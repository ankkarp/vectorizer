import os
import datetime

from sqlalchemy import create_engine, text
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
import logging

# from starlette.exceptions import HTTPException as StarletteHTTPException

load_dotenv()

logging.basicConfig(level=logging.DEBUG)
# logger = logging.getLogger(__name__)


app = FastAPI()


@app.get("/data")
async def get_all():
    df = pd.read_sql(text(f'SELECT * FROM sportobjects'), con)
    return df.astype(str).to_dict('records')


@app.get("/ids")
async def read_object():
    df = pd.read_sql(text(f'SELECT id FROM sportobjects'), con)
    return df.to_dict()


@app.get("/id/{id}")
async def read_object(id: int):
    object_df = pd.read_sql(
        text(f'SELECT * FROM sportobjects WHERE id = {id}'), con).iloc[0]
    return object_df.to_dict()


# @app.get("/stats/sports")
# async def get_sports():
#     df = pd.read_sql(
#         text(f'''SELECT kinds_of_sports FROM sportobjects'''), con).astype(str)
#     sports = np.hstack(df['kinds_of_sports'].str.split(', ').values)
#     return pd.Series(sports).value_counts().to_dict()

@app.get("/stats/timetable")
async def get_sports():
    df = pd.read_sql(
        text(f'''SELECT start_date_of_construction_reconstruction
             FROM sportobjects'''), con)
    return df.str.replace(r'.+\.', '').value_counts().to_dict()


@app.get("/locs")
async def get_locs():
    df = pd.read_sql(
        text(f'''SELECT COALESCE(yandex_object_coordinate_x, yandex_coordinate_center_x) as x,
             COALESCE(yandex_object_coordinate_y, yandex_y_center_coordinate) as y
             FROM sportobjects'''), con).dropna()
    return df.to_dict('records')


@app.get("/stats/{stat}")
async def get_statistic(stat):
    if stat == 'funding':
        df = pd.read_sql(text(f'''SELECT funding_from_the_federal_budget,
                    funding_from_the_budget_of_the_subject_of_the_federation,
                    funding_from_the_budget_of_the_municipality,
                    funding_from_extrabudgetary_sources
                    FROM sportobjects'''), con)
        return {col: df[col].sum(skipna=True) for col in df.columns}
    elif stat in pd.read_sql(text('SELECT * FROM sportobjects LIMIT 1'), con).columns:
        df = pd.read_sql(
            text(f'SELECT {stat} FROM sportobjects'), con).fillna("Не указано")
        data = df[stat].value_counts(dropna=False).to_dict()
        return data
    else:
        raise HTTPException(
            status_code=404,
            detail="Page not found",
            headers={"X-Error": f"Endpoint {stat} doesn't exit"},
        )


# @app.get()s
