"""Standalone Arrow Flight server inside backend container (Phase-3).

Run via:
    python -m observability.flight_server  # starts on 0.0.0.0:8815 by default
"""
from __future__ import annotations

import os
import logging
from typing import Tuple

import pyarrow as pa
import pyarrow.flight as fl

logger = logging.getLogger(__name__)


class PowerFlightServer(fl.FlightServerBase):
    def __init__(self, location: str):
        super().__init__(location)
        self._datasets = {}

    # ------------------------------------------------------------------
    def do_put(self, descriptor, reader, writer):  # type: ignore
        logger.info("[FLIGHT] Received dataset %s", descriptor.path)
        table = reader.read_all()
        self._datasets[tuple(descriptor.path)] = table
        writer.write_table(table)

    def list_flights(self, context, criteria):  # type: ignore
        for path, tbl in self._datasets.items():
            yield fl.FlightInfo(tbl.schema, fl.FlightDescriptor.for_path(*path), [], tbl.num_rows, tbl.nbytes)

    def do_get(self, context, ticket):  # type: ignore
        path = tuple(ticket.ticket.decode().split("/"))
        tbl = self._datasets.get(path)
        if tbl is None:
            raise KeyError(f"Dataset {path} not found")
        return fl.RecordBatchStream(tbl)


if __name__ == "__main__":
    host = os.getenv("FLIGHT_LISTEN", "0.0.0.0:8815")
    server = PowerFlightServer(host)
    logger.info("[FLIGHT] Starting Arrow Flight server at %s", host)
    server.serve() 