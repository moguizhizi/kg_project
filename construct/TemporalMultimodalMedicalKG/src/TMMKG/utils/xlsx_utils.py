# utils/xlsx_utils.py

import pandas as pd
import zipfile
from xml.etree import ElementTree
import posixpath
from typing import List
from pathlib import Path
from typing import Dict
from openpyxl import load_workbook
from typing import List
import logging
import time

from TMMKG.extractors.dataframe_processor import clean_dataframe
from TMMKG.utils.path_utils import safe_filename

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

XML_NS = {
    "main": "http://schemas.openxmlformats.org/spreadsheetml/2006/main",
    "rel": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "pkg_rel": "http://schemas.openxmlformats.org/package/2006/relationships",
}


def build_column_mapping(
    excel_to_label: Dict[str, str],
    property_ontology: Dict[str, str],
    entity_ontology: Dict[str, str],
    strict: bool = True,
) -> Dict[str, str]:
    """
    Build XLSX column mapping WITHOUT reversing ontology dicts.

    Excel column -> AU_P / AU_Q
    """

    column_mapping: Dict[str, str] = {}

    for excel_col, label in excel_to_label.items():
        label = str(label).strip()
        resolved_label = None

        # 1. 在属性表中顺序查
        for au_code, au_label in property_ontology.items():
            if au_code == label:
                resolved_label = au_label
                break

        # 2. 如果属性没找到，再查实体表
        if resolved_label is None:
            for au_code, au_label in entity_ontology.items():
                if au_code == label:
                    resolved_label = au_label
                    break

        # 3. 处理结果
        if resolved_label:
            column_mapping[excel_col] = resolved_label
        else:
            if strict:
                raise ValueError(
                    f"Cannot resolve column '{excel_col}' "
                    f"(label='{label}') to AU_P or AU_Q"
                )

    return column_mapping


def load_unique_column_fast(xlsx_path: str, sheet_name: str, column_name: str) -> List:
    """
    超大 XLSX 文件快速读取指定列，去空值、去重，返回列表。

    参数:
        xlsx_path: Excel 文件路径
        sheet_name: 需要读取的 sheet 名
        column_name: 指定列名

    返回:
        List: 去空、去重后的列值列表
    """
    # 打开 workbook，read_only 模式
    wb = load_workbook(xlsx_path, read_only=True)
    ws = wb[sheet_name]

    # 找到列索引（从 0 开始）
    col_idx = None
    for i, cell in enumerate(next(ws.iter_rows(min_row=1, max_row=1))):
        if cell.value == column_name:
            col_idx = i
            break
    if col_idx is None:
        raise ValueError(f"列 {column_name} 不存在")

    # 流式读取列
    seen = set()
    unique_values = []
    for row in ws.iter_rows(min_row=2):
        val = row[col_idx].value
        if val is not None and val not in seen:
            seen.add(val)
            unique_values.append(val)

    wb.close()
    return unique_values


def load_unique_column(
    path: str, sheet_name: str, column_name: str, as_list: bool = False
) -> pd.DataFrame | List:
    """
    读取 XLSX 文件中指定列，去空、去重，并返回整理后的结果。

    参数:
        path: Excel 文件路径
        sheet_name: 需要读取的 sheet 名
        column_name: 指定列名
        as_list: 是否返回列表 (默认 False 返回 DataFrame)

    返回:
        去空、去重后的 DataFrame 或列表
    """
    # 只读取指定列
    df = pd.read_excel(
        path, sheet_name=sheet_name, usecols=[column_name], engine="openpyxl"
    )

    # 去掉空值
    df = df.dropna(subset=[column_name])

    # 去重
    df = df.drop_duplicates(subset=[column_name])

    # 重置索引
    df = df.reset_index(drop=True)

    if as_list:
        return df[column_name].tolist()
    return df


def get_xlsx_sheetnames(xlsx_path: str) -> List[str]:
    """
    快速获取 XLSX 文件的所有 sheet 名，不读取数据

    参数:
        xlsx_path: Excel 文件路径

    返回:
        List[str]: sheet 名列表
    """
    with zipfile.ZipFile(xlsx_path) as z:
        wb_xml = z.read("xl/workbook.xml")
        root = ElementTree.fromstring(wb_xml)
        ns = {"ns": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        sheet_names = [s.attrib["name"] for s in root.findall(".//ns:sheet", ns)]
    return sheet_names


def read_xlsx_sheet_head_streaming(
    input_path: str | Path,
    sheet_name: str,
    nrows: int,
) -> pd.DataFrame:
    """
    使用 openpyxl read-only 模式读取指定 sheet 的前 nrows 行。

    这个函数用于调试路径，避免 pandas/openpyxl 为了 nrows 仍然解析
    百万行级 sheet 导致启动很慢。
    """
    wb = load_workbook(input_path, read_only=True, data_only=True)

    try:
        ws = wb[sheet_name]
        row_iter = ws.iter_rows(values_only=True)

        try:
            header = next(row_iter)
        except StopIteration:
            return pd.DataFrame()

        columns = [str(col).strip() if col is not None else "" for col in header]
        rows = []

        for _, row in zip(range(nrows), row_iter):
            rows.append(row)

        return pd.DataFrame(rows, columns=columns, dtype=str)

    finally:
        wb.close()


def get_xlsx_sheet_xml_path(xlsx_path: str | Path, sheet_name: str) -> str:
    """
    从 XLSX zip 结构中定位指定 sheet 对应的 worksheet XML 路径。
    """
    with zipfile.ZipFile(xlsx_path) as z:
        workbook_root = ElementTree.fromstring(z.read("xl/workbook.xml"))
        rels_root = ElementTree.fromstring(z.read("xl/_rels/workbook.xml.rels"))

        sheet_rel_id = None
        for sheet in workbook_root.findall(".//main:sheet", XML_NS):
            if sheet.attrib.get("name") == sheet_name:
                sheet_rel_id = sheet.attrib.get(f"{{{XML_NS['rel']}}}id")
                break

        if sheet_rel_id is None:
            raise ValueError(f"Sheet not found: {sheet_name}")

        for rel in rels_root.findall("pkg_rel:Relationship", XML_NS):
            if rel.attrib.get("Id") == sheet_rel_id:
                target = rel.attrib["Target"]
                if target.startswith("/"):
                    return target.lstrip("/")
                return posixpath.normpath(posixpath.join("xl", target))

    raise ValueError(f"Worksheet XML not found for sheet: {sheet_name}")


def cell_ref_to_index(cell_ref: str) -> int:
    col = ""
    for char in cell_ref:
        if char.isalpha():
            col += char.upper()
        else:
            break

    index = 0
    for char in col:
        index = index * 26 + ord(char) - ord("A") + 1

    return index - 1


def get_cell_raw_value(cell) -> tuple[str | None, str | None]:
    cell_type = cell.attrib.get("t")

    if cell_type == "inlineStr":
        text_parts = [
            node.text or ""
            for node in cell.findall(".//main:t", XML_NS)
        ]
        return "".join(text_parts), cell_type

    value = cell.find("main:v", XML_NS)
    return (value.text if value is not None else None), cell_type


def load_required_shared_strings(
    z: zipfile.ZipFile,
    required_indexes: set[int],
) -> dict[int, str]:
    if not required_indexes or "xl/sharedStrings.xml" not in z.namelist():
        return {}

    shared_strings = {}
    current_index = 0
    max_required_index = max(required_indexes)

    for _, elem in ElementTree.iterparse(z.open("xl/sharedStrings.xml"), events=("end",)):
        if elem.tag.endswith("}si"):
            if current_index in required_indexes:
                shared_strings[current_index] = "".join(
                    node.text or "" for node in elem.findall(".//main:t", XML_NS)
                )

            elem.clear()

            if current_index >= max_required_index and required_indexes <= shared_strings.keys():
                break

            current_index += 1

    return shared_strings


def read_xlsx_sheet_head_zip_streaming(
    input_path: str | Path,
    sheet_name: str,
    nrows: int,
) -> pd.DataFrame:
    """
    直接从 XLSX zip/xml 中读取指定 sheet 的表头和前 nrows 行。

    该路径用于超大 XLSX 的 dry-run 调试，避免 openpyxl 初始化整张
    worksheet 带来的等待。
    """
    rows = []
    shared_indexes = set()
    sheet_xml_path = get_xlsx_sheet_xml_path(input_path, sheet_name)

    with zipfile.ZipFile(input_path) as z:
        for _, elem in ElementTree.iterparse(z.open(sheet_xml_path), events=("end",)):
            if not elem.tag.endswith("}row"):
                continue

            row_values = {}
            for fallback_col_index, cell in enumerate(elem.findall("main:c", XML_NS)):
                cell_ref = cell.attrib.get("r", "")
                value, cell_type = get_cell_raw_value(cell)
                col_index = (
                    cell_ref_to_index(cell_ref) if cell_ref else fallback_col_index
                )

                if cell_type == "s" and value is not None:
                    shared_index = int(value)
                    shared_indexes.add(shared_index)
                    row_values[col_index] = ("shared", shared_index)
                else:
                    row_values[col_index] = ("value", value)

            rows.append(row_values)
            elem.clear()

            if len(rows) >= nrows + 1:
                break

        shared_strings = load_required_shared_strings(z, shared_indexes)

    if not rows:
        return pd.DataFrame()

    header_row = rows[0]
    max_col_index = max(header_row.keys(), default=-1)
    columns = []

    for col_index in range(max_col_index + 1):
        value_type, raw_value = header_row.get(col_index, ("value", ""))
        if value_type == "shared":
            value = shared_strings.get(raw_value, "")
        else:
            value = raw_value or ""
        columns.append(str(value).strip())

    data_rows = []
    for row in rows[1:]:
        values = []
        for col_index in range(len(columns)):
            value_type, raw_value = row.get(col_index, ("value", None))
            if value_type == "shared":
                values.append(shared_strings.get(raw_value, ""))
            else:
                values.append(raw_value)
        data_rows.append(values)

    return pd.DataFrame(data_rows, columns=columns, dtype=str)


def xlsx_to_parquet_dataset(
    input_path: str,
    output_dir: str = None,
    compression="zstd",
    overwrite=False,
    multi_label_keywords: list = None,
    nrows: int | None = None,
    required_columns: set[str] | None = None,
    stop_after_first_valid: bool = False,
) -> Dict[str, str]:
    """
    特性：

    超低内存（逐 sheet）
    自动清洗标签污染
    防 schema 漂移
    高性能 vectorized
    防止 KG 标签爆炸

    nrows:
        调试参数。设置后，每个 sheet 通过 zip/xml streaming 只读取前
        nrows 行，避免完整解析超大 sheet。

    required_columns:
        设置后，缺少这些列的 sheet 会被跳过。

    stop_after_first_valid:
        设置后，在成功转换第一个满足 required_columns 的 sheet 后停止。
    """

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(input_path)

    if output_dir is None:
        output_dir = input_path.parent / f"{input_path.stem}_parquet"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching sheet names (fast)...")

    sheet_names = get_xlsx_sheetnames(str(input_path))

    logger.info(f"Found {len(sheet_names)} sheets")

    paths = {}
    total_start = time.perf_counter()

    for sheet in sheet_names:

        safe_sheet = safe_filename(sheet)
        parquet_path = output_dir / f"{safe_sheet}.parquet"

        if parquet_path.exists() and not overwrite:
            logger.info(f"Skip existing -> {sheet}")
            paths[sheet] = str(parquet_path)
            continue

        logger.info(f"Reading sheet -> {sheet}")
        start = time.perf_counter()

        if nrows is None:
            df = pd.read_excel(
                input_path,
                sheet_name=sheet,
                engine="openpyxl",
                dtype=str,
            )
        else:
            df = read_xlsx_sheet_head_zip_streaming(
                input_path=input_path,
                sheet_name=sheet,
                nrows=nrows,
            )

        df = clean_dataframe(df, multi_label_keywords)

        if required_columns:
            missing_columns = sorted(required_columns - set(df.columns))
            if missing_columns:
                logger.info(
                    "Skip sheet %s, missing required columns: %s",
                    sheet,
                    ", ".join(missing_columns),
                )
                continue

        df.to_parquet(parquet_path, compression=compression, index=False)

        logger.info(
            f"Converted [{sheet}] "
            f"rows={len(df)} "
            f"time={time.perf_counter()-start:.2f}s"
        )

        paths[sheet] = str(parquet_path)

        if stop_after_first_valid:
            logger.info("Stop after first valid sheet: %s", sheet)
            break

    logger.info(f"ALL DONE in {time.perf_counter()-total_start:.2f}s")

    return paths
