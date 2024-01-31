import contextlib
import tempfile
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    Union,
)

import numpy as np
import pandas as pd
import pyarrow
import pyarrow.parquet
from pydantic import ConstrainedStr, StrictStr, validator
from pydantic.typing import Literal
from tenacity import Retrying, retry_if_exception_type, stop_after_delay, wait_fixed

from feast import flags_helper
from feast.data_source import DataSource
from feast.errors import (
    BigQueryJobCancelled,
    BigQueryJobStillRunning,
    EntityDFNotDateTime,
    EntitySQLEmptyResults,
    FeastProviderLoginError,
    InvalidEntityType,
)
from feast.feature_logging import LoggingConfig, LoggingSource
from feast.feature_view import DUMMY_ENTITY_ID, DUMMY_ENTITY_VAL, FeatureView
from feast.infra.offline_stores import offline_utils
from feast.infra.offline_stores.offline_store import (
    OfflineStore,
    RetrievalJob,
    RetrievalMetadata,
)
from feast.infra.registry.base_registry import BaseRegistry
from feast.on_demand_feature_view import OnDemandFeatureView
from feast.repo_config import FeastConfigBaseModel, RepoConfig
from feast.saved_dataset import SavedDatasetStorage
from feast.usage import get_user_agent, log_exceptions_and_usage
from feast.infra.utils.bigquery.query_templates import QueryTemplate

from .bigquery_source import (
    BigQueryLoggingDestination,
    BigQuerySource,
    SavedDatasetBigQueryStorage,
)

try:
    from google.api_core import client_info as http_client_info
    from google.api_core.exceptions import NotFound
    from google.auth.exceptions import DefaultCredentialsError
    from google.cloud import bigquery
    from google.cloud.bigquery import Client, SchemaField, Table
    from google.cloud.bigquery._pandas_helpers import ARROW_SCALAR_IDS_TO_BQ
    from google.cloud.storage import Client as StorageClient

except ImportError as e:
    from feast.errors import FeastExtrasDependencyImportError

    raise FeastExtrasDependencyImportError("gcp", str(e))


def get_http_client_info():
    return http_client_info.ClientInfo(user_agent=get_user_agent())


class BigQueryTableCreateDisposition(ConstrainedStr):
    """Custom constraint for table_create_disposition. To understand more, see:
    https://cloud.google.com/bigquery/docs/reference/rest/v2/Job#JobConfigurationLoad.FIELDS.create_disposition"""

    values = {"CREATE_NEVER", "CREATE_IF_NEEDED"}


class BigQueryOfflineStoreConfig(FeastConfigBaseModel):
    """Offline store config for GCP BigQuery"""

    type: Literal["bigquery"] = "bigquery"
    """ Offline store type selector"""

    dataset: StrictStr = "feast"
    """ (optional) BigQuery Dataset name for temporary tables """

    project_id: Optional[StrictStr] = None
    """ (optional) GCP project name used for the BigQuery offline store """
    billing_project_id: Optional[StrictStr] = None
    """ (optional) GCP project name used to run the bigquery jobs at """
    location: Optional[StrictStr] = None
    """ (optional) GCP location name used for the BigQuery offline store.
    Examples of location names include ``US``, ``EU``, ``us-central1``, ``us-west4``.
    If a location is not specified, the location defaults to the ``US`` multi-regional location.
    For more information on BigQuery data locations see: https://cloud.google.com/bigquery/docs/locations
    """

    gcs_staging_location: Optional[str] = None
    """ (optional) GCS location used for offloading BigQuery results as parquet files."""

    table_create_disposition: Optional[BigQueryTableCreateDisposition] = None
    """ (optional) Specifies whether the job is allowed to create new tables. The default value is CREATE_IF_NEEDED."""

    @validator("billing_project_id")
    def project_id_exists(cls, v, values, **kwargs):
        if v and not values["project_id"]:
            raise ValueError(
                "please specify project_id if billing_project_id is specified"
            )
        return v


class BigQueryOfflineStore(OfflineStore):
    @staticmethod
    @log_exceptions_and_usage(offline_store="bigquery")
    def pull_latest_from_table_or_query(
        config: RepoConfig,
        data_source: DataSource,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        created_timestamp_column: Optional[str],
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        assert isinstance(config.offline_store, BigQueryOfflineStoreConfig)
        assert isinstance(data_source, BigQuerySource)
        from_expression = data_source.get_table_query_string()

        partition_by_join_key_string = ", ".join(join_key_columns)
        if partition_by_join_key_string != "":
            partition_by_join_key_string = (
                "PARTITION BY " + partition_by_join_key_string
            )
        timestamps = [timestamp_field]
        if created_timestamp_column:
            timestamps.append(created_timestamp_column)
        timestamp_desc_string = " DESC, ".join(timestamps) + " DESC"
        field_string = ", ".join(join_key_columns + feature_name_columns + timestamps)
        project_id = (
            config.offline_store.billing_project_id or config.offline_store.project_id
        )
        client = _get_bigquery_client(
            project=project_id,
            location=config.offline_store.location,
        )
        query = f"""
            SELECT
                {field_string}
                {f", {repr(DUMMY_ENTITY_VAL)} AS {DUMMY_ENTITY_ID}" if not join_key_columns else ""}
            FROM (
                SELECT {field_string},
                ROW_NUMBER() OVER({partition_by_join_key_string} ORDER BY {timestamp_desc_string}) AS _feast_row
                FROM {from_expression}
                WHERE {timestamp_field} BETWEEN TIMESTAMP('{start_date}') AND TIMESTAMP('{end_date}')
            )
            WHERE _feast_row = 1
            """

        # When materializing a single feature view, we don't need full feature names. On demand transforms aren't materialized
        return BigQueryRetrievalJob(
            query=query,
            client=client,
            config=config,
            full_feature_names=False,
        )

    @staticmethod
    @log_exceptions_and_usage(offline_store="bigquery")
    def pull_all_from_table_or_query(
        config: RepoConfig,
        data_source: DataSource,
        join_key_columns: List[str],
        feature_name_columns: List[str],
        timestamp_field: str,
        start_date: datetime,
        end_date: datetime,
    ) -> RetrievalJob:
        assert isinstance(config.offline_store, BigQueryOfflineStoreConfig)
        assert isinstance(data_source, BigQuerySource)
        from_expression = data_source.get_table_query_string()
        project_id = (
            config.offline_store.billing_project_id or config.offline_store.project_id
        )
        client = _get_bigquery_client(
            project=project_id,
            location=config.offline_store.location,
        )
        field_string = ", ".join(
            join_key_columns + feature_name_columns + [timestamp_field]
        )
        query = f"""
            SELECT {field_string}
            FROM {from_expression}
            WHERE {timestamp_field} BETWEEN TIMESTAMP('{start_date}') AND TIMESTAMP('{end_date}')
        """
        return BigQueryRetrievalJob(
            query=query,
            client=client,
            config=config,
            full_feature_names=False,
        )

    @staticmethod
    @log_exceptions_and_usage(offline_store="bigquery")
    def get_historical_features(
        config: RepoConfig,
        feature_views: List[FeatureView],
        feature_refs: List[str],
        entity_df: Union[pd.DataFrame, str],
        registry: BaseRegistry,
        project: str,
        full_feature_names: bool = False,
    ) -> RetrievalJob:
        # TODO: Add entity_df validation in order to fail before interacting with BigQuery
        assert isinstance(config.offline_store, BigQueryOfflineStoreConfig)
        for fv in feature_views:
            assert isinstance(fv.batch_source, BigQuerySource)
        project_id = (
            config.offline_store.billing_project_id or config.offline_store.project_id
        )
        client = _get_bigquery_client(
            project=project_id,
            location=config.offline_store.location,
        )

        assert isinstance(config.offline_store, BigQueryOfflineStoreConfig)
        if config.offline_store.billing_project_id:
            dataset_project = str(config.offline_store.project_id)
        else:
            dataset_project = client.project
        table_reference = _get_table_reference_for_new_entity(
            client,
            dataset_project,
            config.offline_store.dataset,
            config.offline_store.location,
        )

        entity_schema = _get_entity_schema(
            client=client,
            entity_df=entity_df,
        )

        entity_df_event_timestamp_col = (
            offline_utils.infer_event_timestamp_from_entity_df(entity_schema)
        )

        entity_df_event_timestamp_range = _get_entity_df_event_timestamp_range(
            entity_df,
            entity_df_event_timestamp_col,
            client,
        )

        @contextlib.contextmanager
        def query_generator() -> Iterator[str]:
            _upload_entity_df(
                client=client,
                table_name=table_reference,
                entity_df=entity_df,
            )

            expected_join_keys = offline_utils.get_expected_join_keys(
                project, feature_views, registry
            )

            offline_utils.assert_expected_columns_in_entity_df(
                entity_schema, expected_join_keys, entity_df_event_timestamp_col
            )

            # Build a query context containing all information required to template the BigQuery SQL query
            query_context = offline_utils.get_feature_view_query_context(
                feature_refs,
                feature_views,
                registry,
                project,
                entity_df_event_timestamp_range,
            )

            # Generate the BigQuery SQL query from the query context
            query = offline_utils.build_point_in_time_query(
                query_context,
                left_table_query_string=table_reference,
                entity_df_event_timestamp_col=entity_df_event_timestamp_col,
                entity_df_columns=entity_schema.keys(),
                query_template=QueryTemplate().MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN,
                full_feature_names=full_feature_names,
            )

            try:
                yield query
            finally:
                # Asynchronously clean up the uploaded Bigquery table, which will expire
                # if cleanup fails
                client.delete_table(table=table_reference, not_found_ok=True)

        return BigQueryRetrievalJob(
            query=query_generator,
            client=client,
            config=config,
            full_feature_names=full_feature_names,
            on_demand_feature_views=OnDemandFeatureView.get_requested_odfvs(
                feature_refs, project, registry
            ),
            metadata=RetrievalMetadata(
                features=feature_refs,
                keys=list(entity_schema.keys() - {entity_df_event_timestamp_col}),
                min_event_timestamp=entity_df_event_timestamp_range[0],
                max_event_timestamp=entity_df_event_timestamp_range[1],
            ),
        )

    @staticmethod
    def write_logged_features(
        config: RepoConfig,
        data: Union[pyarrow.Table, Path],
        source: LoggingSource,
        logging_config: LoggingConfig,
        registry: BaseRegistry,
    ):
        destination = logging_config.destination
        assert isinstance(destination, BigQueryLoggingDestination)
        project_id = (
            config.offline_store.billing_project_id or config.offline_store.project_id
        )
        client = _get_bigquery_client(
            project=project_id,
            location=config.offline_store.location,
        )

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            schema=arrow_schema_to_bq_schema(source.get_schema(registry)),
            create_disposition=config.offline_store.table_create_disposition,
            time_partitioning=bigquery.TimePartitioning(
                type_=bigquery.TimePartitioningType.DAY,
                field=source.get_log_timestamp_column(),
            ),
        )

        if isinstance(data, Path):
            for file in data.iterdir():
                with file.open("rb") as f:
                    client.load_table_from_file(
                        file_obj=f,
                        destination=destination.table,
                        job_config=job_config,
                    ).result()

            return

        with tempfile.TemporaryFile() as parquet_temp_file:
            pyarrow.parquet.write_table(table=data, where=parquet_temp_file)

            parquet_temp_file.seek(0)

            client.load_table_from_file(
                file_obj=parquet_temp_file,
                destination=destination.table,
                job_config=job_config,
            ).result()

    @staticmethod
    def offline_write_batch(
        config: RepoConfig,
        feature_view: FeatureView,
        table: pyarrow.Table,
        progress: Optional[Callable[[int], Any]],
    ):
        assert isinstance(config.offline_store, BigQueryOfflineStoreConfig)
        assert isinstance(feature_view.batch_source, BigQuerySource)

        pa_schema, column_names = offline_utils.get_pyarrow_schema_from_batch_source(
            config, feature_view.batch_source, timestamp_unit="ns"
        )
        if column_names != table.column_names:
            raise ValueError(
                f"The input pyarrow table has schema {table.schema} with the incorrect columns {table.column_names}. "
                f"The schema is expected to be {pa_schema} with the columns (in this exact order) to be {column_names}."
            )

        if table.schema != pa_schema:
            table = table.cast(pa_schema)
        project_id = (
            config.offline_store.billing_project_id or config.offline_store.project_id
        )
        client = _get_bigquery_client(
            project=project_id,
            location=config.offline_store.location,
        )

        job_config = bigquery.LoadJobConfig(
            source_format=bigquery.SourceFormat.PARQUET,
            schema=arrow_schema_to_bq_schema(pa_schema),
            create_disposition=config.offline_store.table_create_disposition,
            write_disposition="WRITE_APPEND",  # Default but included for clarity
        )

        with tempfile.TemporaryFile() as parquet_temp_file:
            pyarrow.parquet.write_table(table=table, where=parquet_temp_file)

            parquet_temp_file.seek(0)

            client.load_table_from_file(
                file_obj=parquet_temp_file,
                destination=feature_view.batch_source.table,
                job_config=job_config,
            ).result()


class BigQueryRetrievalJob(RetrievalJob):
    def __init__(
        self,
        query: Union[str, Callable[[], ContextManager[str]]],
        client: bigquery.Client,
        config: RepoConfig,
        full_feature_names: bool,
        on_demand_feature_views: Optional[List[OnDemandFeatureView]] = None,
        metadata: Optional[RetrievalMetadata] = None,
    ):
        if not isinstance(query, str):
            self._query_generator = query
        else:

            @contextlib.contextmanager
            def query_generator() -> Iterator[str]:
                assert isinstance(query, str)
                yield query

            self._query_generator = query_generator
        self.client = client
        self.config = config
        self._full_feature_names = full_feature_names
        self._on_demand_feature_views = on_demand_feature_views or []
        self._metadata = metadata
        if self.config.offline_store.gcs_staging_location:
            self._gcs_path = (
                self.config.offline_store.gcs_staging_location
                + f"/{self.config.project}/export/"
                + str(uuid.uuid4())
            )
        else:
            self._gcs_path = None

    @property
    def full_feature_names(self) -> bool:
        return self._full_feature_names

    @property
    def on_demand_feature_views(self) -> List[OnDemandFeatureView]:
        return self._on_demand_feature_views

    def _to_df_internal(self, timeout: Optional[int] = None) -> pd.DataFrame:
        with self._query_generator() as query:
            df = self._execute_query(query=query, timeout=timeout).to_dataframe(
                create_bqstorage_client=True
            )
            return df

    def to_sql(self) -> str:
        """Returns the underlying SQL query."""
        with self._query_generator() as query:
            return query

    def to_bigquery(
        self,
        job_config: Optional[bigquery.QueryJobConfig] = None,
        timeout: Optional[int] = 1800,
        retry_cadence: Optional[int] = 10,
    ) -> str:
        """
        Synchronously executes the underlying query and exports the result to a BigQuery table. The
        underlying BigQuery job runs for a limited amount of time (the default is 30 minutes).

        Args:
            job_config (optional): A bigquery.QueryJobConfig to specify options like the destination table, dry run, etc.
            timeout (optional): The time limit of the BigQuery job in seconds. Defaults to 30 minutes.
            retry_cadence (optional): The number of seconds for setting how long the job should checked for completion.

        Returns:
            Returns the destination table name or None if job_config.dry_run is True.
        """

        if not job_config:
            today = date.today().strftime("%Y%m%d")
            rand_id = str(uuid.uuid4())[:7]
            if self.config.offline_store.billing_project_id:
                path = f"{self.config.offline_store.project_id}.{self.config.offline_store.dataset}.historical_{today}_{rand_id}"
            else:
                path = f"{self.client.project}.{self.config.offline_store.dataset}.historical_{today}_{rand_id}"
            job_config = bigquery.QueryJobConfig(destination=path)

        if not job_config.dry_run and self.on_demand_feature_views:
            job = self.client.load_table_from_dataframe(
                self.to_df(), job_config.destination
            )
            job.result()
            print(f"Done writing to '{job_config.destination}'.")
            return str(job_config.destination)

        with self._query_generator() as query:
            dest = job_config.destination
            # because setting destination for scripts is not valid
            # remove destination attribute if provided
            job_config.destination = None
            bq_job = self._execute_query(query, job_config, timeout)

            if not job_config.dry_run:
                config = bq_job.to_api_repr()["configuration"]
                # get temp table created by BQ
                tmp_dest = config["query"]["destinationTable"]
                temp_dest_table = f"{tmp_dest['projectId']}.{tmp_dest['datasetId']}.{tmp_dest['tableId']}"

                # persist temp table
                sql = f"CREATE TABLE `{dest}` AS SELECT * FROM {temp_dest_table}"
                self._execute_query(sql, timeout=timeout)

            print(f"Done writing to '{dest}'.")
            return str(dest)

    def _to_arrow_internal(self, timeout: Optional[int] = None) -> pyarrow.Table:
        with self._query_generator() as query:
            q = self._execute_query(query=query, timeout=timeout)
            assert q
            return q.to_arrow()

    @log_exceptions_and_usage
    def _execute_query(
        self, query, job_config=None, timeout: Optional[int] = None
    ) -> Optional[bigquery.job.query.QueryJob]:
        bq_job = self.client.query(query, job_config=job_config)

        if job_config and job_config.dry_run:
            print(
                "This query will process {} bytes.".format(bq_job.total_bytes_processed)
            )
            return None

        block_until_done(client=self.client, bq_job=bq_job, timeout=timeout or 1800)
        return bq_job

    def persist(
        self,
        storage: SavedDatasetStorage,
        allow_overwrite: Optional[bool] = False,
        timeout: Optional[int] = None,
    ):
        assert isinstance(storage, SavedDatasetBigQueryStorage)

        self.to_bigquery(
            bigquery.QueryJobConfig(destination=storage.bigquery_options.table),
            timeout=timeout,
        )

    @property
    def metadata(self) -> Optional[RetrievalMetadata]:
        return self._metadata

    def supports_remote_storage_export(self) -> bool:
        return self._gcs_path is not None

    def to_remote_storage(self) -> List[str]:
        if not self._gcs_path:
            raise ValueError(
                "gcs_staging_location needs to be specified for the big query "
                "offline store when executing `to_remote_storage()`"
            )

        table = self.to_bigquery()

        job_config = bigquery.job.ExtractJobConfig()
        job_config.destination_format = "PARQUET"

        extract_job = self.client.extract_table(
            table,
            destination_uris=[f"{self._gcs_path}/*.parquet"],
            location=self.config.offline_store.location,
            job_config=job_config,
        )
        extract_job.result()

        bucket: str
        prefix: str
        if self.config.offline_store.billing_project_id:
            storage_client = StorageClient(project=self.config.offline_store.project_id)
        else:
            storage_client = StorageClient(project=self.client.project)
        bucket, prefix = self._gcs_path[len("gs://") :].split("/", 1)
        if prefix.startswith("/"):
            prefix = prefix[1:]

        blobs = storage_client.list_blobs(bucket, prefix=prefix)
        results = []
        for b in blobs:
            results.append(f"gs://{b.bucket.name}/{b.name}")
        return results


def block_until_done(
    client: Client,
    bq_job: Union[bigquery.job.query.QueryJob, bigquery.job.load.LoadJob],
    timeout: int = 1800,
    retry_cadence: float = 1,
):
    """
    Waits for bq_job to finish running, up to a maximum amount of time specified by the timeout parameter (defaulting to 30 minutes).

    Args:
        client: A bigquery.client.Client to monitor the bq_job.
        bq_job: The bigquery.job.QueryJob that blocks until done runnning.
        timeout: An optional number of seconds for setting the time limit of the job.
        retry_cadence: An optional number of seconds for setting how long the job should checked for completion.

    Raises:
        BigQueryJobStillRunning exception if the function has blocked longer than 30 minutes.
        BigQueryJobCancelled exception to signify when that the job has been cancelled (i.e. from timeout or KeyboardInterrupt).
    """

    # For test environments, retry more aggressively
    if flags_helper.is_test():
        retry_cadence = 0.1

    def _wait_until_done(bq_job):
        if client.get_job(bq_job).state in ["PENDING", "RUNNING"]:
            raise BigQueryJobStillRunning(job_id=bq_job.job_id)

    try:
        retryer = Retrying(
            wait=wait_fixed(retry_cadence),
            stop=stop_after_delay(timeout),
            retry=retry_if_exception_type(BigQueryJobStillRunning),
            reraise=True,
        )
        retryer(_wait_until_done, bq_job)

    finally:
        if client.get_job(bq_job).state in ["PENDING", "RUNNING"]:
            client.cancel_job(bq_job.job_id)
            raise BigQueryJobCancelled(job_id=bq_job.job_id)

        # We explicitly set the timeout to None because `google-api-core` changed the default value and
        # breaks downstream libraries.
        # https://github.com/googleapis/python-api-core/issues/479
        if bq_job.exception(timeout=None):
            raise bq_job.exception(timeout=None)


def _get_table_reference_for_new_entity(
    client: Client,
    dataset_project: str,
    dataset_name: str,
    dataset_location: Optional[str],
) -> str:
    """Gets the table_id for the new entity to be uploaded."""

    # First create the BigQuery dataset if it doesn't exist
    dataset = bigquery.Dataset(f"{dataset_project}.{dataset_name}")
    dataset.location = dataset_location if dataset_location else "US"

    try:
        client.get_dataset(dataset.reference)
    except NotFound:
        # Only create the dataset if it does not exist
        client.create_dataset(dataset, exists_ok=True)

    table_name = offline_utils.get_temp_entity_table_name()

    return f"{dataset_project}.{dataset_name}.{table_name}"


def _upload_entity_df(
    client: Client,
    table_name: str,
    entity_df: Union[pd.DataFrame, str],
) -> Table:
    """Uploads a Pandas entity dataframe into a BigQuery table and returns the resulting table"""
    job: Union[bigquery.job.query.QueryJob, bigquery.job.load.LoadJob]

    if isinstance(entity_df, str):
        job = client.query(f"CREATE TABLE `{table_name}` AS ({entity_df})")

    elif isinstance(entity_df, pd.DataFrame):
        # Drop the index so that we don't have unnecessary columns
        entity_df.reset_index(drop=True, inplace=True)
        job = client.load_table_from_dataframe(entity_df, table_name)
    else:
        raise InvalidEntityType(type(entity_df))

    block_until_done(client, job)

    # Ensure that the table expires after some time
    table = client.get_table(table=table_name)
    table.expires = datetime.utcnow() + timedelta(minutes=30)
    client.update_table(table, ["expires"])

    return table


def _get_entity_schema(
    client: Client, entity_df: Union[pd.DataFrame, str]
) -> Dict[str, np.dtype]:
    if isinstance(entity_df, str):
        entity_df_sample = (
            client.query(f"SELECT * FROM ({entity_df}) LIMIT 0").result().to_dataframe()
        )

        entity_schema = dict(zip(entity_df_sample.columns, entity_df_sample.dtypes))
    elif isinstance(entity_df, pd.DataFrame):
        entity_schema = dict(zip(entity_df.columns, entity_df.dtypes))
    else:
        raise InvalidEntityType(type(entity_df))

    return entity_schema


def _get_entity_df_event_timestamp_range(
    entity_df: Union[pd.DataFrame, str],
    entity_df_event_timestamp_col: str,
    client: Client,
) -> Tuple[datetime, datetime]:
    if type(entity_df) is str:
        job = client.query(
            f"SELECT MIN({entity_df_event_timestamp_col}) AS min, MAX({entity_df_event_timestamp_col}) AS max "
            f"FROM ({entity_df})"
        )
        res = next(job.result())
        entity_df_event_timestamp_range = (
            res.get("min"),
            res.get("max"),
        )
        if (
            entity_df_event_timestamp_range[0] is None
            or entity_df_event_timestamp_range[1] is None
        ):
            raise EntitySQLEmptyResults(entity_df)
        if type(entity_df_event_timestamp_range[0]) != datetime:
            raise EntityDFNotDateTime()
    elif isinstance(entity_df, pd.DataFrame):
        entity_df_event_timestamp = entity_df.loc[
            :, entity_df_event_timestamp_col
        ].infer_objects()
        if pd.api.types.is_string_dtype(entity_df_event_timestamp):
            entity_df_event_timestamp = pd.to_datetime(
                entity_df_event_timestamp, utc=True
            )
        entity_df_event_timestamp_range = (
            entity_df_event_timestamp.min().to_pydatetime(),
            entity_df_event_timestamp.max().to_pydatetime(),
        )
    else:
        raise InvalidEntityType(type(entity_df))

    return entity_df_event_timestamp_range


def _get_bigquery_client(
    project: Optional[str] = None, location: Optional[str] = None
) -> bigquery.Client:
    try:
        client = bigquery.Client(
            project=project, location=location, client_info=get_http_client_info()
        )
    except DefaultCredentialsError as e:
        raise FeastProviderLoginError(
            str(e)
            + '\nIt may be necessary to run "gcloud auth application-default login" if you would like to use your '
            "local Google Cloud account"
        )
    except EnvironmentError as e:
        raise FeastProviderLoginError(
            "GCP error: "
            + str(e)
            + "\nIt may be necessary to set a default GCP project by running "
            '"gcloud config set project your-project"'
        )

    return client


def arrow_schema_to_bq_schema(arrow_schema: pyarrow.Schema) -> List[SchemaField]:
    bq_schema = []

    for field in arrow_schema:
        if pyarrow.types.is_list(field.type):
            detected_mode = "REPEATED"
            detected_type = ARROW_SCALAR_IDS_TO_BQ[field.type.value_type.id]
        else:
            detected_mode = "NULLABLE"
            detected_type = ARROW_SCALAR_IDS_TO_BQ[field.type.id]

        bq_schema.append(
            SchemaField(name=field.name, field_type=detected_type, mode=detected_mode)
        )

    return bq_schema
