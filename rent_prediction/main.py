import os
import pandas as pd
from prophet import Prophet


def process_row(row, date_headers):
    values = row.iloc[3:].values
    dates = pd.to_datetime(date_headers, format='%Y-%m', errors='coerce')
    prophet_df = pd.DataFrame({'ds': dates, 'y': values})
    prophet_df = prophet_df.dropna()
    if prophet_df.empty:
        return [None] * 12
    try:
        model = Prophet()
        model.fit(prophet_df)
        future = model.make_future_dataframe(periods=12, freq='ME')
        forecast = model.predict(future)
        return forecast['yhat'][-12:].tolist()
    except Exception as e:
        print(f"Error processing row {row.name}: {e}")
        return [None] * 12


def generate_prediction_labels(last_column):
    last_date = pd.to_datetime(last_column, format='%Y-%m', errors='coerce')
    if pd.isna(last_date):
        raise ValueError(f"Invalid date format in column: {last_column}")
    future_dates = pd.date_range(last_date + pd.offsets.MonthBegin(1), periods=12, freq='MS')
    return [date.strftime('%Y-%m') for date in future_dates]


def process_csv(input_file, output_file):
    data = pd.read_csv(input_file)
    date_headers = data.columns[3:]
    last_column = date_headers[-1]
    columns_for_predictions = generate_prediction_labels(last_column)
    predictions = []
    for _, row in data.iterrows():
        predictions.append(process_row(row, date_headers))
    predictions_df = pd.DataFrame(predictions, columns=columns_for_predictions)
    output_data = pd.concat([data, predictions_df], axis=1)
    output_data.to_csv(output_file, index=False)
    print(f"Processed: {input_file} -> {output_file}")


if __name__ == "__main__":
    input_dir = 'data'
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            input_file = os.path.join(input_dir, filename)
            output_file = os.path.join(output_dir, f"predictions_{filename}")
            process_csv(input_file, output_file)
    print(f"All files processed. Results saved in: {output_dir}")
