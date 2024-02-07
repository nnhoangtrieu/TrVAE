



from datetime import datetime

# Get current date and time
current_datetime = datetime.now()

# Print the current date and time
print("Current date and time:", current_datetime)
formatted_datetime = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
print("Formatted current date and time:", formatted_datetime)
