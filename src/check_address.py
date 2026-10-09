import pymysql
from rapidfuzz.distance import Levenshtein

from config import *

region_type_ids = [2, 3, 4, 5]
region_type_name = ['Prov', 'City', 'District', 'Town']
def form_tree(rows):
    prefixes = [row['full_name'].split(' ') for row in rows]
    address_tree = {}
    leaf_id = region_type_ids[-1] - region_type_ids[0]
    for prefix in prefixes:
        node = address_tree
        leaf_node = None
        for i, name in enumerate(prefix):
            if i == leaf_id:
                if leaf_node is not None:
                    leaf_node.append(name)
            elif i == leaf_id - 1:
                leaf_node = node.setdefault(name, [])
            else:
                node = node.setdefault(name, {})
    return address_tree

def flatten_tree(tree, path=None):
    """把 {a:{b:{c:[d,...]}}} 展开成所有叶子路径的组合。"""
    if path is None:
        path = []
    results = []
    for key, value in tree.items():
        if isinstance(value, dict):
            # 还是中间层，继续往下钻，把 key 加入路径
            results.extend(flatten_tree(value, path + [key,]))
        else:
            # 是 list（叶子），把每个元素和前面的路径拼起来
            for leaf in value:
                results.append(path +[key, leaf])
    return results

def search_database(information):
    with pymysql.connect(**MYSQL_CONFIG) as mysql_connect, mysql_connect.cursor(
        pymysql.cursors.DictCursor
    ) as cursor:
        param_list = [
            (region_ids, getattr(information, k) ) for region_ids, k in zip(region_type_ids, region_type_name)
        ]
        param_list.extend([
            (2, information.City),
            (3, information.Prov),
            (3, information.District)
        ])
        param_list = [item for item in param_list if item[1]]
        placeholder = "(region_type=%s and name like %s)"
        sql = (
            f'select full_name from region where {placeholder}'
            + f' or {placeholder}' * (len(param_list)-1)
        )
        params = [
            param 
            for item in param_list
            for param in item
        ]
        cursor.execute(sql, params)
        rows = cursor.fetchall()
    return rows

def compute_score(origin_str, candidate_str):
    return Levenshtein.distance(origin_str, candidate_str)

def check_address(informations):
    fixed_informations = []
    for information in informations:
        if all(getattr(information, k) == '' for k in region_type_name):
            continue
        rows = search_database(information)
        address_tree = form_tree(rows)
        addresses = flatten_tree(address_tree)
        origin_str = (
            information.Prov+information.City+information.District+information.Town
        )
        min_score = float('inf')
        fixed_address = None
        for address in addresses:
            candidate_str = ''.join(address)
            score = compute_score(origin_str, candidate_str)
            if score < min_score:
                min_score = score
                fixed_address = address

        if fixed_address is None or len(fixed_address) != 4:
            fixed_informations.append(information)
        else:
            fixed_information = Dataformat(
                fixed_address[0],
                fixed_address[1],
                fixed_address[2],
                fixed_address[3],
                information.Detail,
                information.Name,
                information.Phone
            )
            fixed_informations.append(fixed_information)
    return fixed_informations