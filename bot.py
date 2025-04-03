import retro
import cv2
import time

# Inicializar o ambiente do Pong
env = retro.make(game="Pong-Atari2600")
obs = env.reset()

# Definir tamanho da janela (largura, altura)
WINDOW_SIZE = (500, 400)

prev_ball = None
PADDLE_SPEED = 6  # Cada comando move o paddle 6 pixels
TOLERANCIA = 15 # Zona de tolerância para evitar oscilações

def process_frame(frame):
    """Converte o frame para escala de cinza e aplica limiar."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)


    return binary

def detect_objects(frame):
    """Detecta a bola e os paddles no frame processado."""
    contours, _ = cv2.findContours(frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ball = None
    paddles = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        if 1 < w < 5 and 1 < h < 5:  # Bola (pequena)
            ball = (x + w // 2, y + h // 2)
        elif h > 10 and h < 17 and w < 10:  # Paddle (barra)
            paddles.append((x, y, w, h))

    return ball, paddles

def calcular_trajetoria(prev_ball, ball, x_limite=140, y_min=34, y_max=194):
    """Calcula os pontos da trajetória da bola até x_limite, considerando as reflexões nas paredes."""
    x0, y0 = ball
    x1, y1 = prev_ball
    dx, dy = x0 - x1, y0 - y1

    trajetoria = [(x0, y0)]
    x, y = x0, y0

    while x < x_limite:
        if dy == 0:
            trajetoria.append((x_limite, y))
            break

        y_destino = y_max if dy > 0 else y_min
        t_parede = abs((y_destino - y) / dy) if dy != 0 else float('inf')
        x_destino = x + t_parede * dx

        if x_destino > x_limite:
            t_final = (x_limite - x) / dx
            y_final = y + t_final * dy
            trajetoria.append((x_limite, y_final))
            break

        trajetoria.append((x_destino, y_destino))
        x, y = x_destino, y_destino
        dy = -dy  # Inverte direção vertical

    return trajetoria

last_predicted_y = None

# Loop principal
while True:
    processed_frame = process_frame(obs)
    ball, paddles = detect_objects(processed_frame)

    display_frame = cv2.cvtColor(processed_frame, cv2.COLOR_GRAY2BGR)

    # Desenhar os paddles (retângulos vermelhos)
    #for x, y, w, h in paddles:
    #    cv2.rectangle(display_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Desenhar a bola (círculo verde)
    #if ball:
    #   cv2.circle(display_frame, ball, 5, (0, 255, 0), -1)

    # Calcular e desenhar trajetória prevista
    predicted_y = None
    if prev_ball and ball and ball[0] > prev_ball[0]:  # Só prever se estiver indo para a direita
        trajeto = calcular_trajetoria(prev_ball, ball)
        #for x, y in trajeto:
        #    cv2.circle(display_frame, (int(x), int(y)), 2, (255, 0, 0), -1)  # Azul para previsão

        if trajeto:
            predicted_y = trajeto[-1][1]  # Última posição Y prevista
            last_predicted_y = predicted_y

    # Atualizar a posição anterior da bola
    prev_ball = ball

    # Controle automático do paddle
    action = [0] * 9
    if len(paddles) >= 2:
        # Ordenar os paddles pelo valor X (o segundo deve ser o da direita)
        paddles.sort(key=lambda p: p[0])
        _, paddle_y, _, paddle_h = paddles[1]  # Pegamos o paddle da direita corretamente
        center_paddle = paddle_y + paddle_h // 2  # Centro do paddle

        if predicted_y is not None:
            deslocamento_necessario = predicted_y - center_paddle

            # Se a distância for menor que a tolerância, não faz nada
            if abs(deslocamento_necessario) > TOLERANCIA:
                movimentos_necessarios = int(deslocamento_necessario / PADDLE_SPEED)

                # Garantir que só mova quando realmente necessário
                if movimentos_necessarios < 0 and deslocamento_necessario < -TOLERANCIA:
                    action[4] = 1  # Mover para cima
                elif movimentos_necessarios > 0 and deslocamento_necessario > TOLERANCIA:
                    action[5] = 1  # Mover para baixo

            print(f"Center Paddle: {center_paddle}, Predicted Y: {predicted_y}, Deslocamento: {deslocamento_necessario}, Movimentos: {movimentos_necessarios}")
    elif(len(paddles)==1):
        paddle_x, _, _, _ = paddles[0]
        if paddle_x < 80 and last_predicted_y is not None and ball and ball[0] > 120:
            if last_predicted_y < 80:
                action[5] = 1  # Mover para baixo
            else:
                action[4] = 1  # Mover para cima

    # Exibir imagem processada
    resized_frame = cv2.resize(display_frame, WINDOW_SIZE, interpolation=cv2.INTER_LINEAR)
    cv2.imshow("Processed Frame", resized_frame)

    key = cv2.waitKey(10)

    # Controles manuais
    if key == ord('q'):
        break
    elif key == 82:
        action[4] = 1
    elif key == 84:
        action[5] = 1

    # Atualizar jogo
    obs, _, done, _ = env.step(action)

    if done:
        obs = env.reset()

env.close()
cv2.destroyAllWindows()
