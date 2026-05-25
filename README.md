# CplaneLoadGenerator

Примеры запуска

python3 main.py --mode stable - стандартный запуск модели для stable распределения

python3 train.py - дообучение модели на пользовательских данных

python3 main.py --devices 100 - запуск с измененным числом мобильных устройств

python3 main.py --mode stable --devices 100 --model ./trained_model - запуск дообученной модели для stable распределения с измененным числом мобильных устройств
