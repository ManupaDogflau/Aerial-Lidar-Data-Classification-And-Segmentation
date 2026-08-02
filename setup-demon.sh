#!/bin/bash

set -e

echo "========================================="
echo "  Instalación LiDAR Sender"
echo "========================================="

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "Proyecto: $PROJECT_DIR"

#############################################
# Buscar python del entorno
#############################################

PYTHON_BIN=$(which python)

echo "Python: $PYTHON_BIN"

#############################################
# Configurar IP fija de eth0
#############################################

echo
echo "[1/5] Configurando eth0..."

if ! grep -q "interface eth0" /etc/dhcpcd.conf; then

sudo tee -a /etc/dhcpcd.conf >/dev/null <<EOF

interface eth0
static ip_address=192.168.1.102/24

EOF

fi

#############################################
# Servicio de ruta del LiDAR
#############################################

echo
echo "[2/5] Creando ruta exclusiva para el LiDAR..."

cat >/tmp/lidar-route.service <<EOF
[Unit]
Description=Route LiDAR through eth0
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/sbin/ip route replace 192.168.1.200 dev eth0 src 192.168.1.102
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
EOF

sudo mv /tmp/lidar-route.service /etc/systemd/system/

#############################################
# Servicio principal
#############################################

echo
echo "[3/5] Creando servicio LiDAR..."

cat >/tmp/lidar.service <<EOF
[Unit]
Description=LiDAR Sender
After=network-online.target lidar-route.service
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
# Activar servicios
#############################################

echo
echo "[4/5] Activando servicios..."

sudo systemctl daemon-reload

sudo systemctl enable lidar-route.service
sudo systemctl enable lidar.service

#############################################
# Reiniciar red
#############################################

echo
echo "[5/5] Reiniciando red..."

sudo systemctl restart dhcpcd || true

echo
echo "========================================="
echo "Instalación completada."
echo
echo "Reinicia la Raspberry:"
echo
echo "    sudo reboot"
echo
echo "Después puedes comprobar:"
echo
echo "    systemctl status lidar"
echo
echo "    journalctl -u lidar -f"
echo
echo "========================================="