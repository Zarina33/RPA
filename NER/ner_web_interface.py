"""
Веб-интерфейс для тестирования NER-системы (Streamlit)
"""

import streamlit as st
import pandas as pd
from io import BytesIO
import json
from datetime import datetime
from ner_extraction_ollama import NameExtractorOllama
import sys

# Конфигурация страницы
st.set_page_config(
    page_title="NER - Извлечение имен и фамилий",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        margin: 1rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        margin: 1rem 0;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
        border: 2px solid #dee2e6;
        margin: 1rem 0;
    }
    .stat-card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #e7f3ff;
        border: 1px solid #b3d9ff;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_extractor(model_name: str):
    """Инициализация экстрактора с кэшированием"""
    try:
        return NameExtractorOllama(model_name=model_name)
    except Exception as e:
        st.error(f"Ошибка инициализации: {str(e)}")
        return None


def process_single_text(extractor, text: str):
    """Обработка одного текста"""
    try:
        with st.spinner("🔄 Обработка текста..."):
            result = extractor.extract(text)
        return result, None
    except Exception as e:
        return None, str(e)


def process_batch_texts(extractor, texts: list):
    """Пакетная обработка текстов"""
    try:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        results = []
        for i, text in enumerate(texts):
            status_text.text(f"Обработка {i+1}/{len(texts)}...")
            result = extractor.extract(text)
            results.append(result)
            progress_bar.progress((i + 1) / len(texts))
        
        status_text.empty()
        progress_bar.empty()
        return results, None
    except Exception as e:
        return None, str(e)


def main():
    """Основная функция интерфейса"""
    
    # Заголовок
    st.markdown('<div class="main-header">🔍 NER - Извлечение имен и фамилий</div>', 
                unsafe_allow_html=True)
    
    # Боковая панель
    with st.sidebar:
        st.header("⚙️ Настройки")
        
        # Выбор модели
        model_name = st.selectbox(
            "Модель",
            ["qwen2.5:14b", "qwen2.5:7b", "qwen2:14b", "qwen2:7b"],
            index=0,
            help="Выберите модель для извлечения имен"
        )
        
        st.markdown("---")
        
        # Информация о модели
        model_info = {
            "qwen2.5:14b": {"ram": "8-10 GB", "quality": "⭐⭐⭐⭐⭐"},
            "qwen2.5:7b": {"ram": "4-6 GB", "quality": "⭐⭐⭐⭐"},
            "qwen2:14b": {"ram": "8-10 GB", "quality": "⭐⭐⭐⭐"},
            "qwen2:7b": {"ram": "4-6 GB", "quality": "⭐⭐⭐"}
        }
        
        info = model_info.get(model_name, {"ram": "Неизвестно", "quality": "⭐⭐⭐"})
        
        st.markdown("**Информация о модели:**")
        st.markdown(f"- Требуется RAM: `{info['ram']}`")
        st.markdown(f"- Качество: {info['quality']}")
        
        st.markdown("---")
        
        # Статистика сессии
        if 'total_processed' not in st.session_state:
            st.session_state.total_processed = 0
        if 'total_found' not in st.session_state:
            st.session_state.total_found = 0
        
        st.markdown("**📊 Статистика сессии:**")
        st.metric("Обработано текстов", st.session_state.total_processed)
        st.metric("Найдено имен", st.session_state.total_found)
        
        if st.button("🔄 Сбросить статистику"):
            st.session_state.total_processed = 0
            st.session_state.total_found = 0
            st.rerun()
        
        st.markdown("---")
        
        # О системе
        with st.expander("ℹ️ О системе"):
            st.markdown("""
            **Named Entity Recognition (NER)**
            
            Система для извлечения имен и фамилий из текста на русском языке.
            
            **Возможности:**
            - Извлечение имени и фамилии
            - Возврат остального текста
            - Пакетная обработка
            - Загрузка Excel/CSV файлов
            
            **Технологии:**
            - Модель: Qwen 2.5
            - Framework: Ollama
            - Interface: Streamlit
            """)
    
    # Инициализация экстрактора
    extractor = get_extractor(model_name)
    
    if extractor is None:
        st.error("❌ Не удалось инициализировать систему. Убедитесь, что Ollama запущен.")
        st.stop()
    
    # Вкладки
    tab1, tab2, tab3, tab4 = st.tabs([
        "📝 Одиночный текст", 
        "📚 Пакетная обработка", 
        "📊 Загрузка файлов",
        "🧪 Примеры"
    ])
    
    # Вкладка 1: Одиночный текст
    with tab1:
        st.header("Обработка одиночного текста")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            text_input = st.text_area(
                "Введите текст для анализа:",
                height=150,
                placeholder="Например: Меня зовут Иван Петров, я работаю программистом"
            )
        
        with col2:
            st.markdown("**Примеры текстов:**")
            examples = [
                "Меня зовут Иван Петров",
                "Документ подписан Анной Сидоровой",
                "Письмо от Михаила Козлова",
                "Заявление Елены Смирновой"
            ]
            
            for example in examples:
                if st.button(f"📋 {example[:25]}...", key=f"ex_{example}"):
                    st.session_state.example_text = example
                    text_input = example
        
        if 'example_text' in st.session_state:
            text_input = st.session_state.example_text
            del st.session_state.example_text
        
        if st.button("🚀 Извлечь имя и фамилию", type="primary", use_container_width=True):
            if text_input.strip():
                result, error = process_single_text(extractor, text_input)
                
                if error:
                    st.markdown(f'<div class="error-box">❌ Ошибка: {error}</div>', 
                               unsafe_allow_html=True)
                else:
                    st.session_state.total_processed += 1
                    if result['first_name'] and result['last_name']:
                        st.session_state.total_found += 1
                    
                    # Результаты
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown("**👤 Имя:**")
                        if result['first_name']:
                            st.success(result['first_name'])
                        else:
                            st.warning("Не найдено")
                    
                    with col2:
                        st.markdown("**👥 Фамилия:**")
                        if result['last_name']:
                            st.success(result['last_name'])
                        else:
                            st.warning("Не найдено")
                    
                    with col3:
                        st.markdown("**✅ Полное имя:**")
                        if result['first_name'] and result['last_name']:
                            st.success(f"{result['first_name']} {result['last_name']}")
                        else:
                            st.warning("Не найдено")
                    
                    if result['remaining_text']:
                        st.markdown("**📄 Остальной текст:**")
                        st.info(result['remaining_text'])
                    
                    st.markdown('</div>', unsafe_allow_html=True)
                    
                    # JSON результат
                    with st.expander("📋 Показать JSON"):
                        st.json(result)
            else:
                st.warning("⚠️ Введите текст для анализа")
    
    # Вкладка 2: Пакетная обработка
    with tab2:
        st.header("Пакетная обработка текстов")
        
        batch_input = st.text_area(
            "Введите тексты (каждый с новой строки):",
            height=200,
            placeholder="Заявление от Ивана Петрова\nДокумент подписан Анной Сидоровой\nПисьмо от Михаила Козлова"
        )
        
        if st.button("🚀 Обработать все тексты", type="primary", use_container_width=True):
            if batch_input.strip():
                texts = [t.strip() for t in batch_input.split('\n') if t.strip()]
                
                if texts:
                    results, error = process_batch_texts(extractor, texts)
                    
                    if error:
                        st.error(f"❌ Ошибка: {error}")
                    else:
                        st.session_state.total_processed += len(texts)
                        found_count = sum(1 for r in results 
                                        if r['first_name'] and r['last_name'])
                        st.session_state.total_found += found_count
                        
                        # Статистика
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                            st.metric("Всего текстов", len(texts))
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                            st.metric("Найдено имен", found_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col3:
                            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                            st.metric("Не найдено", len(texts) - found_count)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        with col4:
                            st.markdown('<div class="stat-card">', unsafe_allow_html=True)
                            success_rate = (found_count / len(texts) * 100) if texts else 0
                            st.metric("Успешность", f"{success_rate:.0f}%")
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Таблица результатов
                        st.markdown("### 📋 Результаты")
                        
                        results_data = []
                        for i, (text, result) in enumerate(zip(texts, results), 1):
                            results_data.append({
                                "№": i,
                                "Исходный текст": text,
                                "Имя": result['first_name'] or "—",
                                "Фамилия": result['last_name'] or "—",
                                "Полное имя": f"{result['first_name']} {result['last_name']}" 
                                             if result['first_name'] and result['last_name'] else "—",
                                "Остаток": result['remaining_text'][:50] + "..." 
                                          if len(result['remaining_text']) > 50 else result['remaining_text']
                            })
                        
                        df = pd.DataFrame(results_data)
                        st.dataframe(df, use_container_width=True, hide_index=True)
                        
                        # Скачивание результатов
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # CSV
                            csv = df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📥 Скачать CSV",
                                data=csv,
                                file_name=f"ner_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
                        
                        with col2:
                            # JSON
                            json_data = json.dumps(
                                [{"text": t, "result": r} for t, r in zip(texts, results)],
                                ensure_ascii=False,
                                indent=2
                            )
                            st.download_button(
                                label="📥 Скачать JSON",
                                data=json_data,
                                file_name=f"ner_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                                mime="application/json"
                            )
            else:
                st.warning("⚠️ Введите тексты для обработки")
    
    # Вкладка 3: Загрузка файлов
    with tab3:
        st.header("Загрузка и обработка файлов")
        
        uploaded_file = st.file_uploader(
            "Выберите файл (CSV или Excel)",
            type=['csv', 'xlsx', 'xls'],
            help="Загрузите CSV или Excel файл с текстами для обработки"
        )
        
        if uploaded_file is not None:
            try:
                # Загрузка файла
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.success(f"✅ Файл загружен: {uploaded_file.name}")
                st.info(f"📊 Строк: {len(df)}, Колонок: {len(df.columns)}")
                
                # Выбор колонки
                text_column = st.selectbox(
                    "Выберите колонку с текстом:",
                    df.columns.tolist()
                )
                
                # Превью
                st.markdown("### 👀 Превью данных:")
                st.dataframe(df.head(10), use_container_width=True)
                
                if st.button("🚀 Обработать файл", type="primary", use_container_width=True):
                    texts = df[text_column].astype(str).tolist()
                    
                    results, error = process_batch_texts(extractor, texts)
                    
                    if error:
                        st.error(f"❌ Ошибка: {error}")
                    else:
                        # Добавление результатов в DataFrame
                        df['extracted_first_name'] = [r['first_name'] for r in results]
                        df['extracted_last_name'] = [r['last_name'] for r in results]
                        df['extracted_full_name'] = [
                            f"{r['first_name']} {r['last_name']}" 
                            if r['first_name'] and r['last_name'] else None
                            for r in results
                        ]
                        df['remaining_text'] = [r['remaining_text'] for r in results]
                        
                        st.success("✅ Обработка завершена!")
                        
                        # Статистика
                        found_count = df['extracted_full_name'].notna().sum()
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Обработано строк", len(df))
                        with col2:
                            st.metric("Найдено имен", found_count)
                        with col3:
                            success_rate = (found_count / len(df) * 100) if len(df) > 0 else 0
                            st.metric("Успешность", f"{success_rate:.0f}%")
                        
                        # Результаты
                        st.markdown("### 📋 Результаты:")
                        st.dataframe(df, use_container_width=True)
                        
                        # Скачивание
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            # Excel
                            output = BytesIO()
                            with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                df.to_excel(writer, index=False)
                            
                            st.download_button(
                                label="📥 Скачать Excel",
                                data=output.getvalue(),
                                file_name=f"ner_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                            )
                        
                        with col2:
                            # CSV
                            csv = df.to_csv(index=False, encoding='utf-8-sig')
                            st.download_button(
                                label="📥 Скачать CSV",
                                data=csv,
                                file_name=f"ner_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
            
            except Exception as e:
                st.error(f"❌ Ошибка при обработке файла: {str(e)}")
    
    # Вкладка 4: Примеры
    with tab4:
        st.header("🧪 Примеры использования")
        
        example_texts = [
            {
                "category": "Формальные документы",
                "texts": [
                    "Заявление от Ивана Петрова о переводе средств на сумму 10000 рублей",
                    "Документ подписан Анной Сидоровой 15 марта 2024 года",
                    "Заявка от Михаила Александровича Иванова о закрытии счета"
                ]
            },
            {
                "category": "Электронные письма",
                "texts": [
                    "От: Елена Смирнова <elena@example.com>\nТема: Встреча завтра",
                    "Письмо от Дмитрия Козлова по вопросу проекта",
                    "Сообщение от Ольги Васильевой получено вчера"
                ]
            },
            {
                "category": "Неформальные тексты",
                "texts": [
                    "Привет! Это Сергей Николаев пишет. Как дела?",
                    "Меня зовут Мария Петрова, я работаю менеджером",
                    "Звонил Алексей Иванов, просил перезвонить"
                ]
            }
        ]
        
        for example_group in example_texts:
            st.markdown(f"### 📂 {example_group['category']}")
            
            for i, text in enumerate(example_group['texts'], 1):
                with st.expander(f"Пример {i}: {text[:50]}..."):
                    st.markdown(f"**Текст:**")
                    st.code(text)
                    
                    if st.button(f"🔍 Обработать", key=f"ex_{example_group['category']}_{i}"):
                        result, error = process_single_text(extractor, text)
                        
                        if error:
                            st.error(f"Ошибка: {error}")
                        else:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.markdown("**Результат:**")
                                st.write(f"- Имя: **{result['first_name'] or 'не найдено'}**")
                                st.write(f"- Фамилия: **{result['last_name'] or 'не найдено'}**")
                            
                            with col2:
                                if result['remaining_text']:
                                    st.markdown("**Остаток:**")
                                    st.info(result['remaining_text'])


if __name__ == "__main__":
    main()

