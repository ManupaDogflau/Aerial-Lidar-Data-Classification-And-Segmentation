#!/bin/bash

set -e

echo "========================================="
echo " Instalación LiDAR Sender"
echo "========================================="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN=$(which python)

echo "Proyecto: $PROJECT_DIR"
echo "Python  : $PYTHON_BIN"

#############################################
# Comprobar NetworkManager
#############################################

if ! systemctl is-active --quiet NetworkManager; then
    echo "ERROR: NetworkManager no está activo."
    exit 1
fi

#############################################
# Configurar eth0
#############################################

echo
echo "[1/4] Configurando eth0..."

ETH_CONN=$(nmcli -t -f NAME,DEVICE connection show | awk -F: '$2=="eth0"{print $1}')

if [ -z "$ETH_CONN" ]; then
    echo "No se encontró la conexión de eth0."
    exit 1
fi

sudo nmcli connection modify "$ETH_CONN" \
    ipv4.method manual \
    ipv4.addresses "192.168.1.102/24" \
    ipv4.gateway "" \
    ipv4.dns "" \
    ipv4.routes "192.168.1.200/32"

#############################################
# Configurar wlan0
#############################################

echo
echo "[2/4] Configurando wlan0..."

WIFI_CONN=$(nmcli -t -f NAME,DEVICE connection show | awk -F: '$2=="wlan0"{print $1}')

if [ -n "$WIFI_CONN" ]; then
    sudo nmcli connection modify "$WIFI_CONN" \
        ipv4.method auto

    # Ruta permanente hacia el portátil
    sudo nmcli connection modify "$WIFI_CONN" \
        +ipv4.routes "192.168.1.143/32"
fi

echo "Aplicando configuración..."

sudo nmcli connection up "$ETH_CONN"

if [ -n "$WIFI_CONN" ]; then
    sudo nmcli connection up "$WIFI_CONN"
fi

#############################################
# Crear servicio
#############################################

echo
echo "[3/4] Creando servicio systemd..."

cat >/tmp/lidar.service <<EOF
[Unit]
Description=LiDAR Sender
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=root
WorkingDirectory=$PROJECT_DIR
ExecStart=$PYTHON_BIN $PROJECT_DIR/main.py

Restart=always
RestartSec=5

StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/lidar.service /etc/systemd/system/

#############################################
# Activar servicio
#############################################

echo
echo "[4/4] Activando servicio..."

sudo systemctl daemon-reload
sudo systemctl enable lidar.service

echo
echo "========================================="
echo "Instalación completada."
echo
echo "Reinicia la Raspberry:"
echo
echo "    sudo reboot"
echo
echo "Comprobar estado:"
echo
echo "    systemctl status lidar"
echo
echo "Ver el log:"
echo
echo "    journalctl -u lidar -f"
echo
echo "========================================="