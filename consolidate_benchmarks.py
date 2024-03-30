#!/usr/bin/env python3

import os
import pandas as pd
import sys
import xlsxwriter
from xlsxwriter.utility import xl_rowcol_to_cell


def write_benchmarks(worksheet, df, columns):
    center_h = workbook.add_format({"align": "center"})
    center_h_bold = workbook.add_format({"align": "center", "bold": True})
    center_h_bold_2places = workbook.add_format(
        {"align": "center", "bold": True, "num_format": "0.00"}
    )
    center_hv = workbook.add_format({"align": "center", "valign": "vcenter"})

    # "Parameter set",
    # "Memory allocation",
    # "Implementation",
    # "Variant",
    # "Operation",
    # "Cycle count",

    benchmark_names = [
        ("Key generation", "crypto_kem_keypair"),
        ("Encapsulation", "crypto_kem_enc"),
        ("Decapsulation", "crypto_kem_dec"),
        ("Polynomial multiplication", "poly_Rq_mul"),
    ]

    impls = {
        "hps2048509": {"NG21": {"amx": "Ours", "neon": "NG21"}},
        "hps2048677": {
            "NG21": {"amx": None, "neon": "NG21"},
            "CCHY23": {
                "amx": "Ours",
                "tc": "CCHY23 tc",
                "tmvp": "CCHY23 tmvp",
            },
        },
        "hps4096821": {"NG21": {"amx": "Ours", "neon": "NG21"}},
        "hrss701": {
            "NG21": {"amx": None, "neon": "NG21"},
            "CCHY23": {"amx": "Ours", "tmvp": "CCHY23 tmvp"},
        },
    }

    for i in range(3):
        worksheet.merge_range(0, i, 1, i, columns[i], center_hv)

    worksheet.merge_range(0, 3, 0, 6, "Operation", center_h)

    for i, names in enumerate(benchmark_names):
        worksheet.write(1, 3 + i, names[0], center_h)

    current_row = 2
    for pset in sorted(df["Parameter set"].unique()):
        merge_pset_start = current_row

        df2 = df[df["Parameter set"] == pset]

        for memalloc in sorted(df2["Memory allocation"].unique()):
            merge_memalloc_start = current_row

            df3 = df2[df2["Memory allocation"] == memalloc]

            for work in sorted(df3["Implementation"].unique()):
                df4 = df3[df3["Implementation"] == work]

                for variant in sorted(df4["Variant"].unique()):
                    impl = impls[pset][work][variant]
                    if not impl:
                        continue

                    worksheet.write(current_row, 2, impl, center_h)

                    df5 = df4[df4["Variant"] == variant]

                    for name, op in benchmark_names:
                        cycle_count = df5[df5["Operation"] == op]["Cycle count"]

                        if not cycle_count.empty:
                            worksheet.write(
                                current_row,
                                3 + benchmark_names.index((name, op)),
                                int(cycle_count.iloc[0]),
                                center_h,
                            )

                    current_row += 1

            worksheet.merge_range(
                merge_memalloc_start, 1, current_row - 1, 1, memalloc, center_hv
            )

        worksheet.merge_range(current_row, 1, current_row, 2, "Speedup", center_h_bold)

        for j in range(3, 7):
            formula = (
                "{=ROUND(MIN(IF("
                f"{xl_rowcol_to_cell(merge_pset_start, 2)}:"
                f'{xl_rowcol_to_cell(current_row - 1, 2)}<>"Ours",'
                f"{xl_rowcol_to_cell(merge_pset_start, j)}:"
                f'{xl_rowcol_to_cell(current_row - 1, j)},""'
                "))/MIN(IF("
                f"{xl_rowcol_to_cell(merge_pset_start, 2)}:"
                f'{xl_rowcol_to_cell(current_row - 1, 2)}="Ours",'
                f"{xl_rowcol_to_cell(merge_pset_start, j)}:"
                f"{xl_rowcol_to_cell(current_row - 1, j)},"
                '"")),2)}'
            )

            worksheet.write(current_row, j, formula, center_h_bold_2places)

        worksheet.merge_range(merge_pset_start, 0, current_row, 0, pset, center_hv)

        current_row += 1

    worksheet.autofit()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        print("Usage: python consolidate_benchmarks.py <CPU name>")
        sys.exit(1)

    path = os.path.join(os.getcwd(), f"speed_results_{sys.argv[1]}")

    if not os.path.isdir(path):
        print(f"Directory {path} does not exist.")
        sys.exit(1)

    all_files = [
        f
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and ":" in f and f.endswith(".txt")
    ]

    data = []

    for file in all_files:
        row = file.split(".txt")[0].split(":")[1:]

        with open(os.path.join(path, file), "r", encoding="utf-8") as f:
            for line in f.read().splitlines():
                if line:
                    data.append(row + line.split(":"))

    columns = [
        "Parameter set",
        "Memory allocation",
        "Implementation",
        "Variant",
        "Operation",
        "Cycle count",
    ]

    df = pd.DataFrame(data, columns=columns).sort_values(by=columns[:-2])

    workbook = xlsxwriter.Workbook(os.path.join(path, "benchmarks.xlsx"))
    worksheet = workbook.add_worksheet("Benchmarks")

    write_benchmarks(worksheet, df, columns)

    workbook.close()
