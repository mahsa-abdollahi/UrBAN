This README.txt file was generated on 2024-06-01 by Mahsa Abdollahi.
 
Abstract: We present a multimodal dataset obtained from a honey bee colony in Montréal, Quebec, Canada, spanning the years of 2021 to 2022. This apiary comprised 10 beehives, with microphones recording more than 2000 hours of high quality raw audio, and also sensors capturing temperature, and humidity. Periodic hive inspections involved monitoring colony honey bee population changes, assessing queen-related conditions, and documenting overall hive health. Additionally, health metrics, such as Varroa mite infestation rates and winter mortality assessments were recorded, offering valuable insights into factors affecting hive health status and resilience.


--------------------
GENERAL INFORMATION
--------------------

1. Title of Dataset: UrBAN: Urban Beehive Acoustics and PheNotyping Dataset

2. Author Information

	A. Principal Investigator Contact Information
		Name: Mahsa Abdollahi
		Institution: INRS
		Email: mahsa.abdollahi@inrs.ca

	B. Associate or Co-investigator Contact Information
		Name: Yi Zhu
		Institution: INRS
		Email: yi.zhu@inrs.ca

	C. Associate or Co-investigator Contact Information
		Name: Heitor Guimarães
		Institution: INRS
		Email: heitor.guimaraes@inrs.ca

         D. Associate or Co-investigator Contact Information
		Name: Nico Coallier
		Institution: Nectar Technologies Inc.
		Email: nico@nectar.buzz

         E. Associate or Co-investigator Contact Information
		Name: Ségolène Maucourt
		Institution: Université Laval
		Email: segolene.maucourt.1@ulaval.ca

         F. Associate or Co-investigator Contact Information
		Name: Pierre Giovenazzo
		Institution: Université Laval 
		Email: pierre.giovenazzo@bio.ulaval.ca

         G. Associate or Co-investigator Contact Information
		Name: Tiago H. Falk
		Institution: INRS
		Email: Tiago.Falk@inrs.ca


3. Date of data collection (single date, range, approximate date): 2021-06-01:2022-10-31

4. Geographic location of data collection: Nectar Technologies Inc., Montreal, QC, Canada.


----------------------------------
SHARING/ACCESS INFORMATION - DATA
----------------------------------

1. Licenses/restrictions placed on the data: 

These data are available under a CC BY 4.0 license <https://creativecommons.org/licenses/by/4.0/> 

2. Links to publications that cite or use the data: -

3. Links/relationships to ancillary data sets or software packages: -

4. Was data derived from another source? no

5. Recommended citation for this dataset:

Abdollahi, M., Zhu, Y., Guimarães, H., Coallier, N., Maucourt, S., Giovenazzo, P., Falk, T. (2024). UrBAN: Urban Beehive Acoustics and PheNotyping Dataset. Federated Research Data Repository. https://doi.org/10.20383/103.0972

---------------------
DATA & FILE OVERVIEW
---------------------

1. File List

   A. Filename: data/temperature_humidity/sensor_2021.csv       
      Short description: This csv files contains temperature and humidity recording from inside the hives. 

   B. Filename: data/audio/beehives_2021   
      Short description: This folder contains 18 tar.gz files. Once you decompress these files, there will be 7011 .wav audio files related to the experiment in 2021. Each recording is named based on the date, time, and a unique hive id (example: "17-08-2021_09h15_HIVE-3628.wav"). The sampling rate of these audio files is 16 kHz. 

   C. Filename: data/audio/beehives_2022   
      Short description: This folder contains 131 tar.gz files. Once you decompress these files, there will be 49965 .wav audio files related to the experiment in 2022. Each recording is named based on the date, time, and a unique hive id  (example: "17-06-2022_13h45_HIVE-3631.wav"). The sampling rate of these audio files is 16 kHz.   

   D. Filename: data/annotations/inspections_2021.csv  
      Short description: This csv file provides the labels for sensor data and audio. It contains information on the number of frames of bees and queen status for each beehive in 2021 (more information is in below sections). 

   E. Filename: data/annotations/inspections_2022.csv   
      Short description: This csv file provides the labels for sensor data and audio. It contains information on the number of frames of bees and queen status for each beehive in 2022(more information is in below sections). 

   F. Filename: data/weather_info/weather_2021_2022.csv   
      Short description: This csv file provides the weather information on the location of the experiments. This data includes temperature, humidity, wind speed and the amount of precipitation (extracted from https://climate.weather.gc.ca/). 


2. Relationship between files, if important: -

3. Additional related data collected that was not included in the current data package: -

4. Are there multiple versions of the dataset? no

-----------------------------------------------------------------
DATA-SPECIFIC INFORMATION FOR: sensor_2021.csv
-----------------------------------------------------------------
This sections provide the structure of sensor_2021.csv file. The column names and their descriptions are as follows:

	Date: Time stamp of the sensor reading (YYYY-MM-DD HH:MM).
	Tag number: Unique identifier for each hive.
	Temperature: Internal temperature in degrees Celsius.
	Humidity: Internal humidity in percentage.

-----------------------------------------------------------------
DATA-SPECIFIC INFORMATION FOR: inspections_2021.csv
-----------------------------------------------------------------
This sections provide the structure of inspections_2021.csv file. The column names and their descriptions are as follows:

	Date: Time stamp of the inspection (YYYY-MM-DD).
	Tag number: Unique identifier for each hive.
	Colony size: Number of boxes for each beehive.
	Fob 1st: Number of frames of bees in the first box.
	Fob 2nd: Number of frames of bees in the second box.
	Fob 3rd: Number of frames of bees in the third box.
	Fob brood: Number of frames of brood.
	Frames of honey: Number of frames of honey.
	Queen status: QR/QNS (queen seen or not seen).
	Open: Time stamp indicating the opening of the box for inspections (HH).
	Close: Time stamp indicating the closing of the box after inspections (HH).
	Note: Additional observations such as whether the beehive was weak or aggressive.


-----------------------------------------------------------------
DATA-SPECIFIC INFORMATION FOR: inspections_2022.csv
-----------------------------------------------------------------
This sections provide the structure of inspections_2022.csv file. The column names and their descriptions are as follows:

	Date: Time stamp of the inspection (YYYY-MM-DD HH:MM).
	Tag number: Unique identifier for each hive.
	Category: Type of inspection data, including:
		Hive grading (e.g., 'strong', 'medium', 'weak')
		Hive status (e.g., 'queenright', 'queenless', 'deadout')
		Frames of bees: Number of frames of bees
		Varroa: Varroa mite measurement
		Treatment (e.g., 'mite away')
		Feeding (e.g., 'sugar')
		Custom practice (e.g., 'add entrance reducer', 'supering')
		Queen management (e.g., 'potential breeder')
		Hive issues (e.g., 'chalk brood')

	Action detail: Detailed description of each category.
	Queen status: Queenright/queenless.
	Is alive: 0/1 indicator (0 indicates a dead hive).
	Report notes: Additional observations such as whether the beehive was weak or aggressive.


-----------------------------------------------------------------
DATA-SPECIFIC INFORMATION FOR: weather_2021_2022.csv
-----------------------------------------------------------------
This sections provide the structure of weather_2021_2022.csv file. The column names and their descriptions are as follows:

	Date/Time (LST): Time stamp of the weather reading (YYYY-MM-DD HH).
	Temp (°C): External temperature in degrees Celsius.
	Rel Hum (%): External humidity in percentage.
	Wind Spd (km/h): Wind speed in kilometers per hour.
	Precip. Amount (mm): Amount of precipitation in millimeters.

-----------------------------------------------------------------
CODE-SPECIFIC INFORMATION FOR: ## ADD FILENAME or DIRECTORY NAME
-----------------------------------------------------------------
Examples of how to use this data using Python are available at our GitHub repository:

https://github.com/Massi331/Urban-Beehive-Acoustics-and-PheNotyping-Dataset
