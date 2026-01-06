#!/bin/bash
# Автоматический перезапуск парсера когда сборщик URL завершится

cd /Users/admin/PycharmProjects/real-estate-valuation

LOG_DIR="data/raw"
RESTART_INTERVAL=3600  # Перезапуск парсера каждый час для подхвата новых URL

echo "$(date): Запуск мониторинга..."

while true; do
    # Проверяем работает ли сборщик URL
    COLLECTOR_PID=$(pgrep -f "krisha_kz.*collect-only")

    if [ -z "$COLLECTOR_PID" ]; then
        echo "$(date): Сборщик URL завершил работу"

        # Останавливаем текущий парсер
        pkill -f "krisha_kz.*--resume"
        sleep 5

        # Запускаем финальный парсер со всеми URL
        echo "$(date): Запуск финального парсера..."
        caffeinate -i python -u -m src.scrapers.krisha_kz --city astana --selenium --resume \
            >> "$LOG_DIR/krisha_astana_scraper.log" 2>&1 &

        echo "$(date): Финальный парсер запущен. Мониторинг завершён."
        break
    fi

    # Проверяем нужен ли перезапуск парсера для подхвата новых URL
    PARSER_PID=$(pgrep -f "krisha_kz.*astana.*--resume")

    if [ -n "$PARSER_PID" ]; then
        # Получаем время работы парсера
        PARSER_START=$(ps -o lstart= -p $PARSER_PID 2>/dev/null)
        if [ -n "$PARSER_START" ]; then
            PARSER_UPTIME=$(( $(date +%s) - $(date -j -f "%c" "$PARSER_START" +%s 2>/dev/null || echo 0) ))

            if [ "$PARSER_UPTIME" -gt "$RESTART_INTERVAL" ]; then
                echo "$(date): Перезапуск парсера для подхвата новых URL..."
                pkill -f "krisha_kz.*--resume"
                sleep 5
                caffeinate -i python -u -m src.scrapers.krisha_kz --city astana --selenium --resume \
                    >> "$LOG_DIR/krisha_astana_scraper.log" 2>&1 &
                echo "$(date): Парсер перезапущен"
            fi
        fi
    else
        # Парсер не работает, запускаем
        echo "$(date): Парсер не найден, запускаем..."
        caffeinate -i python -u -m src.scrapers.krisha_kz --city astana --selenium --resume \
            >> "$LOG_DIR/krisha_astana_scraper.log" 2>&1 &
    fi

    # Статистика
    TOTAL_URLS=$(sqlite3 "$LOG_DIR/krisha_kz_astana_urls.db" "SELECT COUNT(*) FROM urls;" 2>/dev/null)
    PARSED_URLS=$(sqlite3 "$LOG_DIR/krisha_kz_astana_urls.db" "SELECT SUM(parsed) FROM urls;" 2>/dev/null)
    echo "$(date): URL в БД: $TOTAL_URLS, спарсено: $PARSED_URLS"

    sleep 300  # Проверка каждые 5 минут
done
