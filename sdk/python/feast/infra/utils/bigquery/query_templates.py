class QueryTemplate():
    # TODO: Optimizations
    #   * Use GENERATE_UUID() instead of ROW_NUMBER(), or join on entity columns directly
    #   * Precompute ROW_NUMBER() so that it doesn't have to be recomputed for every query on entity_dataframe
    #   * Create temporary tables instead of keeping all tables in memory

    # Note: Keep this in sync with sdk/python/feast/infra/offline_stores/redshift.py:MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN

    MULTIPLE_FEATURE_VIEW_POINT_IN_TIME_JOIN = """
    /*
    Compute a deterministic hash for the `left_table_query_string` that will be used throughout
    all the logic as the field to GROUP BY the data
    */
    CREATE TEMP TABLE entity_dataframe AS (
        SELECT *,
            {{entity_df_event_timestamp_col}} AS entity_timestamp
            {% for featureview in featureviews %}
                {% if featureview.entities %}
                ,CONCAT(
                    {% for entity in featureview.entities %}
                        CAST({{entity}} AS STRING),
                    {% endfor %}
                    CAST({{entity_df_event_timestamp_col}} AS STRING)
                ) AS {{featureview.name}}__entity_row_unique_id
                {% else %}
                ,CAST({{entity_df_event_timestamp_col}} AS STRING) AS {{featureview.name}}__entity_row_unique_id
                {% endif %}
            {% endfor %}
        FROM `{{ left_table_query_string }}`
    );

    {% for featureview in featureviews %}
    CREATE TEMP TABLE {{ featureview.name }}__cleaned AS (
        WITH {{ featureview.name }}__entity_dataframe AS (
            SELECT
                {{ featureview.entities | join(', ')}}{% if featureview.entities %},{% else %}{% endif %}
                entity_timestamp,
                {{featureview.name}}__entity_row_unique_id
            FROM entity_dataframe
            GROUP BY
                {{ featureview.entities | join(', ')}}{% if featureview.entities %},{% else %}{% endif %}
                entity_timestamp,
                {{featureview.name}}__entity_row_unique_id
        ),

        /*
        This query template performs the point-in-time correctness join for a single feature set table
        to the provided entity table.

        1. We first join the current feature_view to the entity dataframe that has been passed.
        This JOIN has the following logic:
            - For each row of the entity dataframe, only keep the rows where the `timestamp_field`
            is less than the one provided in the entity dataframe
            - If there a TTL for the current feature_view, also keep the rows where the `timestamp_field`
            is higher the the one provided minus the TTL
            - For each row, Join on the entity key and retrieve the `entity_row_unique_id` that has been
            computed previously

        The output of this CTE will contain all the necessary information and already filtered out most
        of the data that is not relevant.
        */

        {{ featureview.name }}__subquery AS (
            SELECT
                {{ featureview.timestamp_field }} as event_timestamp,
                {{ featureview.created_timestamp_column ~ ' as created_timestamp,' if featureview.created_timestamp_column else '' }}
                {{ featureview.entity_selections | join(', ')}}{% if featureview.entity_selections %},{% else %}{% endif %}
                {% for feature in featureview.features %}
                    {{ feature }} as {% if full_feature_names %}{{ featureview.name }}__{{featureview.field_mapping.get(feature, feature)}}{% else %}{{ featureview.field_mapping.get(feature, feature) }}{% endif %}{% if loop.last %}{% else %}, {% endif %}
                {% endfor %}
            FROM {{ featureview.table_subquery }}
            WHERE {{ featureview.timestamp_field }} <= '{{ featureview.max_event_timestamp }}'
            {% if featureview.ttl == 0 %}{% else %}
            AND {{ featureview.timestamp_field }} >= '{{ featureview.min_event_timestamp }}'
            {% endif %}
        ),

        {{ featureview.name }}__base AS (
            SELECT
                subquery.*,
                entity_dataframe.entity_timestamp,
                entity_dataframe.{{featureview.name}}__entity_row_unique_id
            FROM {{ featureview.name }}__subquery AS subquery
            INNER JOIN {{ featureview.name }}__entity_dataframe AS entity_dataframe
            ON TRUE
                AND subquery.event_timestamp <= entity_dataframe.entity_timestamp

                {% if featureview.ttl == 0 %}{% else %}
                AND subquery.event_timestamp >= Timestamp_sub(entity_dataframe.entity_timestamp, interval {{ featureview.ttl }} second)
                {% endif %}

                {% for entity in featureview.entities %}
                AND subquery.{{ entity }} = entity_dataframe.{{ entity }}
                {% endfor %}
        ),

        /*
        2. If the `created_timestamp_column` has been set, we need to
        deduplicate the data first. This is done by calculating the
        `MAX(created_at_timestamp)` for each event_timestamp.
        We then join the data on the next CTE
        */
        {% if featureview.created_timestamp_column %}
        {{ featureview.name }}__dedup AS (
            SELECT
                {{featureview.name}}__entity_row_unique_id,
                event_timestamp,
                MAX(created_timestamp) as created_timestamp
            FROM {{ featureview.name }}__base
            GROUP BY {{featureview.name}}__entity_row_unique_id, event_timestamp
        ),
        {% endif %}

        /*
        3. The data has been filtered during the first CTE "*__base"
        Thus we only need to compute the latest timestamp of each feature.
        */
        {{ featureview.name }}__latest AS (
        SELECT
            event_timestamp,
            {% if featureview.created_timestamp_column %}created_timestamp,{% endif %}
            {{featureview.name}}__entity_row_unique_id
        FROM
        (
            SELECT *,
                ROW_NUMBER() OVER(
                    PARTITION BY {{featureview.name}}__entity_row_unique_id
                    ORDER BY event_timestamp DESC{% if featureview.created_timestamp_column %},created_timestamp DESC{% endif %}
                ) AS row_number
            FROM {{ featureview.name }}__base
            {% if featureview.created_timestamp_column %}
                INNER JOIN {{ featureview.name }}__dedup
                USING ({{featureview.name}}__entity_row_unique_id, event_timestamp, created_timestamp)
            {% endif %}
        )
        WHERE row_number = 1
    )

    /*
    4. Once we know the latest value of each feature for a given timestamp,
    we can join again the data back to the original "base" dataset
    */

        SELECT base.*
        FROM {{ featureview.name }}__base as base
        INNER JOIN {{ featureview.name }}__latest
        USING(
            {{featureview.name}}__entity_row_unique_id,
            event_timestamp
            {% if featureview.created_timestamp_column %}
                ,created_timestamp
            {% endif %}
        )
    );


    {% endfor %}
    /*
    Joins the outputs of multiple time travel joins to a single table.
    The entity_dataframe dataset being our source of truth here.
    */

    SELECT {{ final_output_feature_names | join(', ')}}
    FROM entity_dataframe
    {% for featureview in featureviews %}
    LEFT JOIN (
        SELECT
            {{featureview.name}}__entity_row_unique_id
            {% for feature in featureview.features %}
                ,{% if full_feature_names %}{{ featureview.name }}__{{featureview.field_mapping.get(feature, feature)}}{% else %}{{ featureview.field_mapping.get(feature, feature) }}{% endif %}
            {% endfor %}
        FROM {{ featureview.name }}__cleaned
    ) USING ({{featureview.name}}__entity_row_unique_id)
    {% endfor %}
    """
