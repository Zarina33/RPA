#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт для извлечения колонок CreditCode и Name из словаря кодов назначения
"""

import pandas as pd
from pathlib import Path
import sys

def extract_codes_to_txt():
    """Извлекает CreditCode и Name из Excel и сохраняет в текстовый файл"""
    
    # Путь к входному файлу
    input_file = Path(__file__).parent / 'docs' / 'Словарь кодов назначения.xlsx'
    
    # Путь к выходному файлу
    output_file = Path(__file__).parent / 'purpose_codes.txt'
    
    # Проверка существования файла
    if not input_file.exists():
        print(f"✗ Файл не найден: {input_file}")
        sys.exit(1)
    
    try:
        # Загрузка данных из Excel
        print(f"Загрузка данных из {input_file.name}...")
        df = pd.read_excel(input_file)
        
        print(f"✓ Данные успешно загружены: {len(df)} записей\n")
        
        # Проверка наличия нужных колонок
        if 'CreditCode' not in df.columns or 'Name' not in df.columns:
            print("✗ Ошибка: В файле не найдены колонки 'CreditCode' и/или 'Name'")
            print(f"Доступные колонки: {', '.join(df.columns)}")
            sys.exit(1)
        
        # Извлечение нужных колонок
        result_df = df[['CreditCode', 'Name']]
        
        # Удаление строк с пустыми значениями
        result_df = result_df.dropna()
        
        print(f"Извлечено записей (после удаления пустых): {len(result_df)}")
        print(f"\nПример данных:")
        print(result_df.head(10).to_string(index=False))
        print()
        
        # Сохранение в текстовый файл
        with open(output_file, 'w', encoding='utf-8') as f:
            # Заголовок
            f.write("="*80 + "\n")
            f.write("СЛОВАРЬ КОДОВ НАЗНАЧЕНИЯ\n")
            f.write("CreditCode и Name\n")
            f.write("="*80 + "\n\n")
            
            # Заголовки колонок
            f.write(f"{'CreditCode':<15} {'Name':<60}\n")
            f.write("-"*80 + "\n")
            
            # Данные
            for _, row in result_df.iterrows():
                # Конвертируем CreditCode в int, чтобы убрать .0
                credit_code = str(int(row['CreditCode'])).strip()
                name = str(row['Name']).strip()
                f.write(f"{credit_code:<15} {name:<60}\n")
            
            # Итог
            f.write("\n" + "="*80 + "\n")
            f.write(f"Всего записей: {len(result_df)}\n")
            f.write("="*80 + "\n")
        
        print(f"✓ Данные успешно сохранены в файл: {output_file}")
        print(f"  Всего записей в файле: {len(result_df)}")
        
    except Exception as e:
        print(f"✗ Ошибка при обработке файла: {e}")
        sys.exit(1)

if __name__ == "__main__":
    extract_codes_to_txt()

