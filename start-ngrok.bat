@echo off
echo Starting ngrok tunnel for Telegram webhook...
echo.
echo Tunnel URL: https://cityless-proactively-fermina.ngrok-free.dev
echo Target: localhost:8443
echo.

ngrok http --domain=cityless-proactively-fermina.ngrok-free.dev 8443
