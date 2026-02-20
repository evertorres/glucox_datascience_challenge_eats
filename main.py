
import pandas as pd
from utils import unpack_json_column, generate_nudges_measurement_df, add_cumulative_counts, mark_responses, mean_responses, plot_nudge_response, chi_test_per_number_of_nudges


def main():
    print('Starting...')
    print('Opening patient registry file...')

    try: 
        patient_registry = pd.read_csv('data/patient_registry.csv')
        
        if not patient_registry.empty: 
            print('Patient registry file read successfully')

    except Exception as e:
        print(f"Cannot read patient registry file. Error: {e}")
        return
   
    print('\nOpening Log file...')

    try: 
        app_logs = pd.read_json('data/app_logs.jsonl', lines=True)
        
        if not app_logs.empty: 
            print('App logs file read successfully')

    except Exception as e:
        print(f"Cannot read app logs file. Error: {e}")
        return

    print('\nUnpacking Log Column...')

    try: 
        unpacked = unpack_json_column(app_logs, 'payload')
        print('Unpacking successful')
        print(unpacked.head())
    except Exception as e:
        print(f"Error unpacking JSON column: {e}")
        return

    print('\nCreating nudges and measurements dataframes...')

    try:
        df_nudges, df_meas = generate_nudges_measurement_df(unpacked)
        df_nudges = df_nudges.sort_values(by='timestamp')
        df_meas = df_meas.sort_values(by=['timestamp'])
        print('Nudges and measurement dataframes created successfully')
    except Exception as e:
        print(f"Error creating nudges and measurement dataframes: {e}")
        return

    print(df_nudges.head())
    print(df_meas.head())
    print('\nCalculating cumulative counts for nudges...')

    try:
        df_nudges = add_cumulative_counts(
        df=df_nudges, 
        group_col='patient_id', 
        type_col='nudge_type', 
        prefix='cumulative'
    )
    except Exception as e:
        print(f"Error calculating cumulative counts for nudges: {e}")
        return

    print(df_nudges.head())

    print('\nCalculating assertive nudges...')
    
    try: 
        pd_asof = pd.merge_asof(df_meas, df_nudges, on='timestamp', by='patient_id', direction='backward', tolerance=pd.Timedelta(hours=4))
        print('Assertive nudges calculated successfully')
        print(pd_asof.head())
    except Exception as e:
        print(f"Error calculating assertive nudges: {e}")
        return

    print('\nMarking response in nudges...')

    try:
        df_final = mark_responses(df_nudges, pd_asof, target_col='event_id', source_col='event_id_y', new_column='has_response')
        print(df_final.head())
    except Exception as e:
        print(f"Error marking response in nudges: {e}")
        return

    print('\nMerging patients registry and final nudges...')

    try:
        df_patients_nudges = df_final.merge(patient_registry, left_on='patient_id', right_on='patient_id')
        print(df_patients_nudges.head())
    except Exception as e:
        print(f"Error merging patients registry and final nudges: {e}")
        return

    print('\nCalculating mean response rates per gentle reminder group, age group and risk segment...')

    ## Reporting: 

    with open("report/report.txt", "w") as file:
        file.write("Reporting data\\n")
        file.write("Ever Augusto Torres Silva")
        file.write("Castor Technical test")

    print('\nCalculating mean responses for Gentle Reminder group...')

    try:
        df_gentle = mean_responses(df_patients_nudges, 'Gentle_Reminder')
        print(df_gentle.head())

        with open("report/report.txt", "a") as file:
            file.write('\nMean Responses for Gentle Reminders')
            file.write('\n' + df_gentle.to_string())

    except Exception as e:
        print(f"Error calculating mean response rates: {e}")


    print('\nCalculating mean responses for Urgent Alert group...')

    try:
        df_urgent = mean_responses(df_patients_nudges, 'Urgent_Alert')
        print(df_urgent.head())

        with open("report/report.txt", "a") as file:
            file.write('\nMean Responses for Urgent Alerts')
            file.write('\n' + df_urgent.to_string())

    except Exception as e:
        print(f"Error calculating mean response rates: {e}")

        
    print('\nPlotting results...')
    try:
        plot_nudge_response(df_patients_nudges, filename="response_global.png")

    except Exception as e:
        print(f"Error plotting results: {e}")

    print('\nPlotting results per age group...')

    try:
        plot_nudge_response(df_patients_nudges, group_col="age_group", filename="response_age_group.png")

    except Exception as e:
        print(f"Error plotting results: {e}")

    print('\nPlotting results per risk segment group...')

    try:
        plot_nudge_response(df_patients_nudges, group_col="risk_segment", filename="response_risk_segment.png")

    except Exception as e:
        print(f"Error plotting results: {e}") 

    print('\nCalculating significant difference between Nudge 1 and subsequent nudges - Global...')

    try:
        output = chi_test_per_number_of_nudges(df_patients_nudges)
        with open("report/report.txt", "a") as file:
            file.write('\nCalculating significant difference between Nudge 1 and subsequent nudges - Global')
            file.write('\n' + output)
    except Exception as e:
        print(f'Error generating stats report: {e}')
            
if __name__ == "__main__":
    main()
