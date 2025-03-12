SELECT 
    f.Source,
    f.Destination,
    f.Class,
    f.Total_stops,
    f.Airline,
    f.Fare AS actual_fare,
    min_fare_data.min_fare,
    f.Journey_day AS actual_journeyday,
    min_fare_data.journeyday_of_minimumfare
FROM flight_data f
JOIN (
    -- Subquery to get minimum fare and its corresponding journey day
    SELECT DISTINCT ON (Source, Destination, Class, Total_stops, Airline)
        Source,
        Destination,
        Class,
        Total_stops,
        Airline,
        Fare AS min_fare,
        Journey_day AS journeyday_of_minimumfare
    FROM flight_data
    ORDER BY Source, Destination, Class, Total_stops, Airline, Fare ASC
) AS min_fare_data
ON f.Source = min_fare_data.Source
AND f.Destination = min_fare_data.Destination
AND f.Class = min_fare_data.Class
AND f.Total_stops = min_fare_data.Total_stops
AND f.Airline = min_fare_data.Airline;
