import random

jogador = {'hp': 100,
           'ataque': 10,
           'sala_atual': 'entrada'}

mapa = {'entrada': {
    'descricao': 'Você está na entrada de uma masmorra escura e úmida. Um corredor segue para o norte.',
    'saidas': {'norte': 'corredor'}},
    'corredor': {
    'descricao': 'Você está em um corredor estreito. Há uma porta a leste e a entrada da masmorra fica ao sul.',
    'saidas': {'sul': 'entrada', 'leste': 'tesouro'}},
    'tesouro': {
    'descricao': 'Você encontrou uma sala com um baú brilhante! Mas espere... um zumbi te ataca!',
    'saidas': {'oeste': 'corredor'},
    'monstro': {'nome': 'Zumbi', 'hp': 50, 'ataque': 5}}}

is_running = True


def mostrar_status(jogador):
    print(f"HP: {jogador['hp']:.2f}")
    print(f"Ataque: {jogador['ataque']}")
    print(f"Sala atual: {jogador['sala_atual']}")


def mostrar_sala(mapa, jogador):
    sala = mapa[jogador['sala_atual']]
    print(sala['descricao'])


def mover_jogador(mapa, jogador, direcao):
    if direcao in mapa[jogador['sala_atual']]['saidas']:
        jogador['sala_atual'] = mapa[jogador['sala_atual']]['saidas'][direcao]
        print(f'Você entrou para {direcao}')
    else:
        print('Movimento inválido.')


def iniciar_combate(jogador, monstro):
    while jogador['hp'] > 0 and monstro['hp'] > 0:
        move = input('O que deseja fazer?\n1. Atacar\n2. Esquivar\n3. Fugir\n')
        if move == '1':
            monstro['hp'] -= jogador['ataque']
            jogador['hp'] -= monstro['ataque']
            print(
                f"Você atacou o {monstro['nome']} e causou {jogador['ataque']} de dano.")
            print(
                f"{monstro['nome']} atacou você e causou {monstro['ataque']:.2f} de dano.")
        elif move == '2':
            if random.choice([True, False]):
                print('Você se esquivou do ataque.')
            else:
                print('Você não conseguiu se esquivar do ataque.')
                jogador['hp'] -= monstro['ataque']
                print(
                    f"{monstro['nome']} atacou você e causou {monstro['ataque']:.2f} de dano.")
        elif move == '3':
            print('Você fugiu do combate.')
            jogador['sala_atual'] = 'corredor'
            monstro['ataque'] *= 1.5
            return True
        else:
            print('Movimento inválido.')
        print()
        mostrar_status(jogador)
        print()
        monstro['ataque'] *= 1.2
    if monstro['hp'] <= 0:
        print('Parabens você matou o monstro e abriu o Tesouro!')
        return False
    else:
        print('Você foi derrotado. Game Over!')
        return False


def main():

    round = 1
    while is_running:
        print(f'Rodada: {round}')
        mostrar_status(jogador)
        mostrar_sala(mapa, jogador)
        print()
        direcao = input('Para onde deseja seguir? ')
        mover_jogador(mapa, jogador, direcao.lower())
        print()
        if jogador['sala_atual'] == 'tesouro':
            mostrar_sala(mapa, jogador)
            print()
            is_running = iniciar_combate(jogador, mapa['tesouro']['monstro'])
        print()

        round += 1


if __name__ == '__main__':
    main()
