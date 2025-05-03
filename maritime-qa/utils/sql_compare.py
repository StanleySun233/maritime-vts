import pandas as pd
import numpy as np
from typing import Union, List, Tuple
from utils.db_util import query

def compare_sql_results(gt_result: Union[pd.DataFrame, str], gp_result: Union[pd.DataFrame, str]) -> Tuple[float, str]:
    """
    比较两个SQL查询结果并计算匹配分数
    
    参数:
        gt_result: 真实SQL查询结果（DataFrame或"ERROR SQL"字符串）
        gp_result: 预测SQL查询结果（DataFrame或"ERROR SQL"字符串）
        
    返回:
        Tuple[float, str]: 匹配分数（范围0-1）和备注信息
    """
    # 检查预测SQL是否有错误
    if isinstance(gp_result, str) and gp_result == "ERROR SQL":
        return 0.0, "预测SQL执行错误"
    
    # 检查真实SQL是否有错误
    if isinstance(gt_result, str) and gt_result == "ERROR SQL":
        return 0.0, "真实SQL执行错误"
    
    # 确保结果是DataFrame
    if not isinstance(gt_result, pd.DataFrame) or not isinstance(gp_result, pd.DataFrame):
        return 0.0, "结果不是DataFrame类型"
    
    # 将列名转换为小写
    gt_result.columns = gt_result.columns.str.lower()
    gp_result.columns = gp_result.columns.str.lower()
    
    # 处理重复列名
    def handle_duplicate_columns(df):
        # 检查是否有重复列名
        if len(df.columns) != len(set(df.columns)):
            # 获取重复的列名
            duplicate_cols = df.columns[df.columns.duplicated()].tolist()
            # 为重复列添加后缀
            for i, col in enumerate(df.columns):
                if col in duplicate_cols:
                    # 计算该列名出现的次数
                    count = list(df.columns[:i]).count(col)
                    # 重命名列
                    df.rename(columns={col: f"{col}_{count}"}, inplace=True)
        return df
    
    try:
        # 处理重复列名
        gt_result = handle_duplicate_columns(gt_result)
        gp_result = handle_duplicate_columns(gp_result)
        
        # 如果两个DataFrame都只有一列，直接比较值
        if len(gt_result.columns) == 1 and len(gp_result.columns) == 1:
            # 获取唯一的列名
            gt_col = gt_result.columns[0]
            gp_col = gp_result.columns[0]
            
            # 将两个列的值转换为字符串并创建集合
            gt_values = set(gt_result[gt_col].astype(str))
            gp_values = set(gp_result[gp_col].astype(str))
            
            # 计算匹配的值的数量
            matched = len(gt_values.intersection(gp_values))
            
            # 计算基础分数
            base_score = matched / len(gt_values)
            
            # 如果预测结果多于真实结果，应用惩罚因子
            if len(gp_values) > len(gt_values):
                # 计算多余项的比例
                extra_ratio = (len(gp_values) - len(gt_values)) / len(gt_values)
                # 惩罚因子：随着多余项比例的增加而减小
                penalty = 1 / (1 + extra_ratio)
                return base_score * penalty, "单列比较，预测结果多于真实结果"
            
            return base_score, "单列比较"
        
        # 对列名进行排序
        gt_result = gt_result.reindex(sorted(gt_result.columns), axis=1)
        gp_result = gp_result.reindex(sorted(gp_result.columns), axis=1)
        
        # 对两个DataFrame进行去重
        gt_result = gt_result.drop_duplicates()
        gp_result = gp_result.drop_duplicates()
        
        # 获取列名
        gt_columns = set(gt_result.columns)
        gp_columns = set(gp_result.columns)
        
        # 检查是否有name或mmsi列
        has_name_or_mmsi = ('name' in gt_columns or 'mmsi' in gt_columns) and ('name' in gp_columns or 'mmsi' in gp_columns)
        
        # 如果真实查询结果为空
        if len(gt_result) == 0:
            if len(gp_result) == 0:
                return 1.0, "两者都为空"  # 两者都为空，返回1分
            else:
                return 0.0, "真实结果为空但预测结果不为空"  # 真实为空但预测不为空，返回0分
        
        # 处理数字列，四舍五入到小数点后两位
        def process_numeric_columns(df):
            for col in df.columns:
                # 检查列是否为数值类型
                if pd.api.types.is_numeric_dtype(df[col]):
                    # 对数值列进行四舍五入到小数点后两位
                    df[col] = df[col].apply(lambda x: round(float(x), 2) if pd.notna(x) else x)
            return df
        
        gt_result = process_numeric_columns(gt_result)
        gp_result = process_numeric_columns(gp_result)
        
        # 如果有name或mmsi列，优先使用这些列进行匹配
        if has_name_or_mmsi:
            # 确定匹配列
            match_column = None
            if 'mmsi' in gt_columns and 'mmsi' in gp_columns:
                match_column = 'mmsi'
            elif 'name' in gt_columns and 'name' in gp_columns:
                match_column = 'name'
            
            if match_column:
                # 计算匹配分数
                gt_values = set(gt_result[match_column].astype(str))
                gp_values = set(gp_result[match_column].astype(str))
                matched = len(gt_values.intersection(gp_values))
                
                # 计算基础分数
                base_score = matched / len(gt_values)
                
                # 如果预测结果多于真实结果，应用惩罚因子
                if len(gp_values) > len(gt_values):
                    # 计算多余项的比例
                    extra_ratio = (len(gp_values) - len(gt_values)) / len(gt_values)
                    # 惩罚因子：随着多余项比例的增加而减小
                    penalty = 1 / (1 + extra_ratio)
                    return base_score * penalty, f"使用{match_column}列匹配，预测结果多于真实结果"
                
                return base_score, f"使用{match_column}列匹配"
        
        # 如果列名完全一致
        if gt_columns == gp_columns:
            # 将DataFrame转换为字符串表示进行比较，不考虑顺序
            gt_str = gt_result.astype(str).apply(lambda x: ','.join(sorted(x)), axis=1)
            gp_str = gp_result.astype(str).apply(lambda x: ','.join(sorted(x)), axis=1)
            
            gt_values = set(gt_str)
            gp_values = set(gp_str)
            matched = len(gt_values.intersection(gp_values))
            
            # 计算基础分数
            base_score = matched / len(gt_values)
            
            # 如果预测结果多于真实结果，应用惩罚因子
            if len(gp_values) > len(gt_values):
                # 计算多余项的比例
                extra_ratio = (len(gp_values) - len(gt_values)) / len(gt_values)
                # 惩罚因子：随着多余项比例的增加而减小
                penalty = 1 / (1 + extra_ratio)
                return base_score * penalty, "列名完全一致，预测结果多于真实结果"
            
            return base_score, "列名完全一致"
        
        # 如果有重叠列
        common_columns = gt_columns.intersection(gp_columns)
        if common_columns:
            # 使用重叠列进行比较
            gt_subset = gt_result[list(common_columns)]
            gp_subset = gp_result[list(common_columns)]
            
            # 将DataFrame转换为字符串表示进行比较，不考虑顺序
            gt_str = gt_subset.astype(str).apply(lambda x: ','.join(sorted(x)), axis=1)
            gp_str = gp_subset.astype(str).apply(lambda x: ','.join(sorted(x)), axis=1)
            
            gt_values = set(gt_str)
            gp_values = set(gp_str)
            matched = len(gt_values.intersection(gp_values))
            
            # 计算基础分数
            base_score = matched / len(gt_values)
            
            # 如果预测结果多于真实结果，应用惩罚因子
            if len(gp_values) > len(gt_values):
                # 计算多余项的比例
                extra_ratio = (len(gp_values) - len(gt_values)) / len(gt_values)
                # 惩罚因子：随着多余项比例的增加而减小
                penalty = 1 / (1 + extra_ratio)
                return base_score * penalty, f"使用重叠列{list(common_columns)}匹配，预测结果多于真实结果"
            
            return base_score, f"使用重叠列{list(common_columns)}匹配"
        
        # 如果完全没有重叠列
        return 0.0, "完全没有重叠列"
    except Exception as e:
        return 0.0, f"比较过程中发生错误: {str(e)}"

def execute_and_compare(gt_sql: str, gp_sql: str) -> Tuple[float, str]:
    """
    执行SQL查询并比较结果
    
    参数:
        gt_sql: 真实SQL查询语句
        gp_sql: 预测SQL查询语句
        
    返回:
        Tuple[float, str]: 匹配分数（范围0-1）和备注信息
    """
    # 执行SQL查询
    gt_result = query(gt_sql)
    gp_result = query(gp_sql)
    
    # 比较结果
    return compare_sql_results(gt_result, gp_result)

# 测试函数
if __name__ == "__main__":
    # 测试用例1：直接比较DataFrame
    gt_df = pd.DataFrame({
        'mmsi': [123, 456],
        'name': ['船A', '船B'],
        'latitude': [30.5, 31.2],
        'longitude': [120.3, 121.5]
    })
    
    gp_df = pd.DataFrame({
        'longitude': [120.3, 121.5],
        'latitude': [30.5, 31.2],
        'name': ['船A', '船B'],
        'mmsi': [123, 456]
    })
    
    score1, remark1 = compare_sql_results(gt_df, gp_df)
    print(f"直接比较DataFrame的匹配分数: {score1}, 备注: {remark1}")
    
    # 测试用例2：执行SQL查询并比较
    gt_sql = "SELECT mmsi, name, latitude, longitude FROM ships LIMIT 10"
    gp_sql = "SELECT mmsi, name, latitude, longitude FROM ships LIMIT 5"
    
    score2, remark2 = execute_and_compare(gt_sql, gp_sql)
    print(f"执行SQL查询并比较的匹配分数: {score2}, 备注: {remark2}") 