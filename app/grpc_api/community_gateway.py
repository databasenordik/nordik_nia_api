from functools import wraps

from app.execution.filters import apply_predicates
from app.planning.plan_schema import FilterOperator, Predicate


# These gateway methods all take scope, file_ids, predicates as their first
# three arguments. Enforce the UI selection at the data boundary, independently
# of whether the planner includes a community in its generated plan.
_PREDICATE_METHODS = {
    "count_records", "count_records_by_file", "group_values", "group_values_multi",
    "field_stats", "duplicate_values", "record_completeness", "interval_rows",
    "interval_stats", "list_records", "list_records_window", "sample_records",
    "sample_records_page",
}


class CommunityGateway:
    def __init__(self, gateway, catalog, communities):
        self._gateway = gateway
        self._catalog = catalog
        self._predicates = [Predicate(
            field="community", operator=FilterOperator.IN, value=list(communities)
        )]

    def __getattr__(self, name):
        method = getattr(self._gateway, name)
        if name not in _PREDICATE_METHODS:
            return method

        @wraps(method)
        async def filtered(scope, file_ids, predicates, *args, **kwargs):
            return await method(scope, file_ids, [*self._predicates, *predicates], *args, **kwargs)

        return filtered

    async def retrieve_candidates(self, scope, file_ids, query, method, limit=20):
        return await self._gateway.retrieve_candidates(
            scope, file_ids, query, method, limit, predicates=self._predicates
        )

    async def get_records_by_source_ids(self, scope, source_row_ids):
        rows = await self._gateway.get_records_by_source_ids(scope, source_row_ids)
        return apply_predicates(rows, self._predicates, self._catalog, scope.allowed_file_ids)

    async def get_raw_fields(self, scope, source_row_ids, keys):
        rows = await self.get_records_by_source_ids(scope, source_row_ids)
        return await self._gateway.get_raw_fields(
            scope, [int(row["source_row_id"]) for row in rows], keys
        )

    async def suggest_name_matches(self, scope, file_ids, query, **kwargs):
        matches = await self._gateway.suggest_name_matches(scope, file_ids, query, **kwargs)
        rows = await self.get_records_by_source_ids(
            scope, [int(match["source_row_id"]) for match in matches]
        )
        allowed = {(int(row["file_id"]), int(row["source_row_id"])) for row in rows}
        return [m for m in matches if (int(m["file_id"]), int(m["source_row_id"])) in allowed]
