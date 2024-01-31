from typing import Optional, Dict, Set, Tuple
import json
from feast import BigQuerySource
from feast.data_source import DataSourceProto
from feast.errors import DataSourceNoNameException
from feast.infra.offline_stores.bigquery_source import BigQueryOptions


class IcebergSource(BigQuerySource):
    """
    IcebergSource is an extension of BigQuerySource that includes additional parameters specific to Iceberg.
    """

    def __init__(
        self,
        *,
        name: Optional[str] = None,
        timestamp_field: Optional[str] = None,
        table: Optional[str] = None,
        created_timestamp_column: Optional[str] = "",
        field_mapping: Optional[Dict[str, str]] = None,
        query: Optional[str] = None,
        description: Optional[str] = "",
        tags: Optional[Dict[str, str]] = None,
        owner: Optional[str] = "",
        eventTypes: Optional[Set[str]] = None,
        dateRange: Optional[Tuple[str, str]] = None,
        isStreaming: Optional[bool] = None,
        useEventTimeAligner: Optional[bool] = None,
        **kwargs,
    ):
        """
        Create an IcebergSource from an existing table or query.

        Args:
            name (optional): Name for the source. Defaults to the table if not specified, in which
                case the table must be specified.
            timestamp_field (optional): Event timestamp field used for point in time
                joins of feature values.
            table (optional): BigQuery table where the features are stored. Exactly one of 'table'
                and 'query' must be specified.
            table (optional): The BigQuery table where features can be found.
            created_timestamp_column (optional): Timestamp column when row was created, used for deduplicating rows.
            field_mapping (optional): A dictionary mapping of column names in this data source to feature names in a feature table
                or view. Only used for feature columns, not entities or timestamp columns.
            query (optional): The query to be executed to obtain the features. Exactly one of 'table'
                and 'query' must be specified.
            description (optional): A human-readable description.
            tags (optional): A dictionary of key-value pairs to store arbitrary metadata.
            owner (optional): The owner of the bigquery source, typically the email of the primary
                maintainer.
            eventTypes (optional): Set of event types to be included.
            dateRange (optional): Tuple of start and end dates for the data range.
            isStreaming (optional): Boolean flag indicating if the source is streaming.
            useEventTimeAligner (optional): Boolean flag indicating if event time aligner is used.
            **kwargs: Other arguments inherited from BigQuerySource.
        """

        if table is None and query is None:
            raise ValueError('No "table" or "query" argument provided.')

        self.bigquery_options = BigQueryOptions(table=table, query=query)

        # If no name, use the table as the default name.
        if name is None and table is None:
            raise DataSourceNoNameException()
        name = name or table
        assert name

        iceberg_options_dict = {
            "eventTypes": eventTypes,
            "dateRange": dateRange,
            "isStreaming": isStreaming,
            "useEventTimeAligner": useEventTimeAligner
        }
        self.iceberg_options = DataSourceProto.CustomSourceOptions(configuration=json.dumps(iceberg_options_dict).encode('utf-8'))

        super().__init__(
            name=name,
            timestamp_field=timestamp_field,
            created_timestamp_column=created_timestamp_column,
            field_mapping=field_mapping,
            description=description,
            tags=tags,
            owner=owner,
            table=table,
            query=query
        )
    
    @staticmethod
    def from_proto(data_source: DataSourceProto):
        """
        Create an IcebergSource from a protobuf representation.

        Args:
            data_source (DataSourceProto): Protobuf representation of a data source.

        Returns:
            IcebergSource: A new IcebergSource object.
        """
        assert data_source.HasField("bigquery_options")
        assert data_source.HasField("custom_options")

        # Decode the iceberg_options to retrieve fields
        iceberg_options = json.loads(data_source.custom_options.decode('utf-8'))
        event_types = iceberg_options.get("eventTypes", [])
        date_range = iceberg_options.get("dateRange", "")
        is_streaming = iceberg_options.get("isStreaming", False)
        use_event_time_aligner = iceberg_options.get("useEventTimeAligner", False)


        return IcebergSource(
            name=data_source.name,
            field_mapping=dict(data_source.field_mapping),
            table=data_source.bigquery_options.table,
            timestamp_field=data_source.timestamp_field,
            created_timestamp_column=data_source.created_timestamp_column,
            query=data_source.bigquery_options.query,
            description=data_source.description,
            tags=dict(data_source.tags),
            owner=data_source.owner,
            eventTypes=event_types,
            dateRange=date_range,
            isStreaming=is_streaming,
            useEventTimeAligner=use_event_time_aligner,
        )

    def to_proto(self) -> DataSourceProto:
        """
        Convert the IcebergSource to its protobuf representation.

        Returns:
            DataSourceProto: Protobuf representation of the IcebergSource.
        """
        return DataSourceProto(
            name=self.name,
            type=DataSourceProto.BATCH_BIGQUERY,
            field_mapping=self.field_mapping,
            bigquery_options=self.bigquery_options.to_proto(),
            custom_options=self.iceberg_options,
            description=self.description,
            tags=self.tags,
            owner=self.owner,
            timestamp_field=self.timestamp_field,
            created_timestamp_column=self.created_timestamp_column,
        )
    