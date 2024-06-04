import json


class FilterUtil:
    """
    Utility to convert MongoDB-style query JSON into SQL.
    Supports a *very* limited subset, only what is used by the tested
    datasets:
     - label match - {"tags": "1"}
     - Conjunction of two matches (AND) - {"$and": [{"tags": "2"}, {"tags": "3"}]
    """

    @staticmethod
    def to_sql(filter: dict) -> str:
        """
        Given a MongoDB-style query, convert to an equivalent SQL WHERE clause.
        """
        if filter is None:
            return ""
        return "WHERE " + FilterUtil._parse_expression(filter)

    @staticmethod
    def _parse_expression(expr: dict) -> str:
        assert len(expr) == 1
        for key, value in expr.items():
            if key == "$and":
                return FilterUtil._parse_and(value)
            else:
                return FilterUtil._parse_match(key, value)
        return ""

    @staticmethod
    def _parse_match(field: str, value: dict) -> str:
        return f"metadata @> '{{\"{field}\": [{json.dumps(value)}]}}'"

    @staticmethod
    def _parse_and(e: dict):
        assert len(e) == 2
        lhs = e[0]
        assert len(lhs) == 1
        rhs = e[1]
        assert len(rhs) == 1
        return (
            FilterUtil._parse_expression(lhs)
            + " AND "
            + FilterUtil._parse_expression(rhs)
        )
