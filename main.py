import streamlit as st
import pandas as pd
import numpy as np
import joblib
from haversine import haversine
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Предсказание цен на дома округа King",
    page_icon="🏠",
    layout="wide"
)


@st.cache_resource
def load_model():
    """Загрузка модели"""
    try:
        model = joblib.load('best_xgb.joblib')
        return model
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None


def create_features(data):
    """Создание признаков, которые были разработаны в Feature Engineering"""
    try:
        data['sqft_living_x_grade'] = data['sqft_living'] * data['grade']

        center = (47.6076, -122.3369)
        data['dist_to_center'] = data.apply(
            lambda row: haversine((row['lat'], row['long']), center),
            axis=1
        )

        data['waterfront_x_grade'] = data['waterfront'] * data['grade']

        data['house_age'] = 2024 - data['yr_built']

        def assign_region(lat, long):
            if lat > 47.6 and long > -122.3:
                return '0'  # Северо-восток
            elif lat <= 47.6 and long > -122.3:
                return '2'  # Юго-восток
            elif lat > 47.6 and long <= -122.3:
                return '3'  # Северо-запад
            else:
                return '1'  # Юго-запад

        data['region'] = data.apply(
            lambda row: assign_region(row['lat'], row['long']), axis=1)

        region_dummies = pd.get_dummies(
            data['region'], prefix='region', drop_first=True)
        data = pd.concat([data, region_dummies], axis=1)

        data['has_basement'] = np.where(data['sqft_basement'] > 0, 1, 0)

        return data
    except Exception as e:
        st.error(f"Ошибка при создании признаков: {e}")
        return None


def predict_price(model, input_data):
    """Предсказание цены дома"""
    try:
        df = pd.DataFrame([input_data])

        df_processed = create_features(df)

        if df_processed is None:
            return None

        if 'region' in df_processed.columns:
            df_processed = df_processed.drop('region', axis=1)

        expected_features = model.feature_names_in_

        for feature in expected_features:
            if feature not in df_processed.columns:
                df_processed[feature] = 0

        df_processed = df_processed[expected_features]

        prediction = model.predict(df_processed)[0]

        return prediction
    except Exception as e:
        st.error(f"Ошибка при предсказании: {e}")
        st.error(f"Детали ошибки: {str(e)}")
        return None


def main():
    st.title("🏠 Предсказание цен на дома округа King")
    st.markdown("Введите характеристики дома для получения предсказания цены")

    model = load_model()
    if model is None:
        st.stop()

    st.sidebar.header("Параметры дома")

    st.sidebar.subheader("🏡 Основные характеристики")
    floors = st.sidebar.slider(
        "Количество этажей", min_value=1.0, max_value=3.5, value=1.0, step=0.5)

    st.sidebar.subheader("📐 Площади")
    sqft_living = st.sidebar.slider(
        "Жилая площадь (кв.фт)", min_value=290, max_value=13540, value=2000)
    sqft_basement = st.sidebar.slider(
        "Площадь подвала (кв.фт)", min_value=0, max_value=4820, value=0)

    st.sidebar.subheader("⭐ Качество и состояние")
    waterfront = st.sidebar.checkbox("Дом на берегу")
    view = st.sidebar.slider(
        "Оценка вида (0-4)", min_value=0, max_value=4, value=0)
    condition = st.sidebar.slider(
        "Состояние дома (1-5)", min_value=1, max_value=5, value=3)
    grade = st.sidebar.slider(
        "Качество и дизайн (1-13)", min_value=1, max_value=13, value=7)

    st.sidebar.subheader("📅 Временные характеристики")
    yr_built = st.sidebar.slider(
        "Год постройки", min_value=1900, max_value=2015, value=1980)

    st.sidebar.subheader("📍 Местоположение")
    zipcode = st.sidebar.number_input(
        "Почтовый индекс", min_value=98001, max_value=98199, value=98103)
    lat = st.sidebar.slider("Широта", min_value=47.1559,
                            max_value=47.7776, value=47.5480, format="%.4f")
    long = st.sidebar.slider("Долгота", min_value=-122.5190,
                             max_value=-121.3154, value=-122.3238, format="%.4f")

    st.sidebar.subheader("🏘️ Соседние дома")
    sqft_living15 = st.sidebar.slider(
        "Средняя жилая площадь 15 соседних домов", min_value=399, max_value=6210, value=2000)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Введенные параметры")

        params_data = {
            "Параметр": [
                "Этажи", "Жилая площадь",
                "Площадь подвала", "На берегу", "Вид", "Состояние",
                "Качество", "Год постройки", "Почтовый индекс",
                "Широта", "Долгота", "Жилая площадь соседей"
            ],
            "Значение": [
                str(floors), str(sqft_living),
                str(sqft_basement), "Да" if waterfront else "Нет", str(
                    view), str(condition),
                str(grade), str(yr_built), str(zipcode),
                f"{lat:.4f}", f"{long:.4f}", str(
                    sqft_living15)
            ]
        }

        params_df = pd.DataFrame(params_data)
        st.dataframe(params_df, use_container_width=True)

    with col2:
        st.subheader("Предсказание")

        if st.button("Предсказать цену", type="primary", use_container_width=True):
            input_data = {
                'sqft_living': sqft_living,
                'floors': floors,
                'waterfront': int(waterfront),
                'view': view,
                'condition': condition,
                'grade': grade,
                'sqft_basement': sqft_basement,
                'yr_built': yr_built,
                'zipcode': zipcode,
                'lat': lat,
                'long': long,
                'sqft_living15': sqft_living15
            }

            with st.spinner("Вычисляю предсказание..."):
                predicted_price = predict_price(model, input_data)

            if predicted_price is not None:
                st.metric(
                    label="Предсказанная цена",
                    value=f"${predicted_price:,.0f}",
                    delta=None
                )

                st.info(
                    f"Цена за квадратный фут: ${predicted_price/sqft_living:.0f}")
            else:
                st.error(
                    "Не удалось выполнить предсказание. Проверьте введенные данные")


if __name__ == "__main__":
    main()
