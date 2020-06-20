from dataclasses import (
    asdict,
    dataclass
)

from sqlalchemy import create_engine
from sqlalchemy.engine.base import Connection


@dataclass
class SQLConfig:
    """Dictionary to hold snowflake credentials"""
    database: str


def make_sql_con(sql_config: SQLConfig) -> Connection:
    """Make string to be used for sql connection """
    sql_engine_template = 'sqlite://{database}.db'
    sql_con_str = sql_engine_template.format(**asdict(sql_config))
    sql_alchemy_con = create_engine(sql_con_str).connect()
    return sql_alchemy_con
