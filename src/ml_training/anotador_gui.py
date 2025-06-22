import tkinter as tk
from tkinter import messagebox
import pandas as pd
import yfinance as yf
# A importação do mplfinance e do backend do Tkinter são essenciais
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class AnotadorApp:
    def __init__(self, root, arquivo_candidatos, arquivo_saida):
        self.root = root
        self.root.title("Ferramenta de Anotação do Garimpeiro")
        self.root.geometry("1200x800")

        self.arquivo_saida = arquivo_saida

        print("Carregando padrões candidatos...")
        try:
            self.df_candidatos = pd.read_csv(arquivo_candidatos, parse_dates=[
                                             'data_inicio', 'data_fim', 'data_cabeca'])
        except FileNotFoundError:
            messagebox.showerror(
                "Erro", f"Arquivo de candidatos '{arquivo_candidatos}' não encontrado!")
            self.root.destroy()
            return
        except ValueError as e:
            if 'data_cabeca' in str(e):
                messagebox.showerror(
                    "Erro de Coluna", f"Não encontrei a coluna 'data_cabeca' no seu CSV. Por favor, adicione-a para marcar a cabeça dos padrões OCO e OCOI.")
            else:
                messagebox.showerror("Erro ao ler CSV", str(e))
            self.root.destroy()
            return

        self.indice_atual = 0
        self.lista_aprovados = []

        # --- Layout da Interface ---
        self.frame_grafico = tk.Frame(root)
        self.frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        self.frame_controles = tk.Frame(root)
        self.frame_controles.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = tk.Label(
            self.frame_controles, text="Carregando...", font=("Arial", 12))
        self.info_label.pack(side=tk.LEFT, padx=10)

        self.rejeitar_btn = tk.Button(
            self.frame_controles, text="❌ Rejeitar", command=self.rejeitar, font=("Arial", 12), bg="salmon")
        self.rejeitar_btn.pack(side=tk.RIGHT, padx=10)

        self.aprovar_btn = tk.Button(self.frame_controles, text="✅ Aprovar",
                                     command=self.aprovar, font=("Arial", 12), bg="lightgreen")
        self.aprovar_btn.pack(side=tk.RIGHT)

        self.canvas = None

        self.carregar_proximo_padrao()

    def plotar_grafico(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        data_inicio_contexto = self.data_inicio - pd.Timedelta(days=5)
        data_fim_contexto = self.data_fim + pd.Timedelta(days=10)

        print(
            f"Buscando dados para o gráfico de {data_inicio_contexto.date()} a {data_fim_contexto.date()}...")
        df_grafico = yf.download(
            tickers='BTC-USD', start=data_inicio_contexto, end=data_fim_contexto, interval='4h')

        if df_grafico.empty:
            print("Não foi possível buscar dados. Pulando...")
            self.root.after(100, self.rejeitar)
            return

        if isinstance(df_grafico.columns, pd.MultiIndex):
            df_grafico.columns = df_grafico.columns.get_level_values(0)

        df_grafico.index = df_grafico.index.tz_localize(None)

        cols_to_convert = ['Open', 'High', 'Low', 'Close', 'Volume']
        for col in cols_to_convert:
            df_grafico[col] = pd.to_numeric(df_grafico[col], errors='coerce')

        df_grafico.dropna(subset=cols_to_convert, inplace=True)

        if df_grafico.empty:
            print("DataFrame ficou vazio após a limpeza. Pulando...")
            self.root.after(100, self.rejeitar)
            return

        df_grafico[cols_to_convert] = df_grafico[cols_to_convert].astype(float)

        fig, axlist = mpf.plot(df_grafico, type='candle',
                               style='charles', returnfig=True)

        ax = axlist[0]
        ax.set_title(f"Analisar Padrão: {self.tipo_padrao}")

        try:
            start_pos = (df_grafico.index -
                         self.data_inicio).to_series().abs().argmin()
            end_pos = (df_grafico.index -
                       self.data_fim).to_series().abs().argmin()
            ax.axvspan(start_pos, end_pos, color='yellow', alpha=0.3)

            # --- AJUSTE PARA OCO E OCOI ---
            # A condição agora verifica se o tipo do padrão está na lista ['OCO', 'OCOI']
            if self.tipo_padrao in ['OCO', 'OCOI'] and self.data_cabeca:
                # -------------------------------
                head_pos = (df_grafico.index -
                            self.data_cabeca).to_series().abs().argmin()
                ax.axvline(x=head_pos, color='dodgerblue',
                           linestyle='--', linewidth=1.2, label='Cabeça (Head)')
                ax.legend()

        except Exception as e:
            print(
                f"Aviso: Não foi possível desenhar a faixa amarela do padrão. Erro: {e}")

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

    def carregar_proximo_padrao(self):
        if self.indice_atual >= len(self.df_candidatos):
            messagebox.showinfo("Fim!", "Todos os padrões foram analisados!")
            self.salvar_e_sair()
            return

        padrao_atual = self.df_candidatos.iloc[self.indice_atual]
        self.tipo_padrao = padrao_atual['tipo_padrao']
        self.data_inicio = padrao_atual['data_inicio']
        self.data_fim = padrao_atual['data_fim']

        self.data_cabeca = None
        if 'data_cabeca' in padrao_atual and pd.notna(padrao_atual['data_cabeca']):
            self.data_cabeca = padrao_atual['data_cabeca']

        self.info_label.config(
            text=f"Padrão {self.indice_atual + 1}/{len(self.df_candidatos)} | Tipo: {self.tipo_padrao} | Início: {self.data_inicio.date()}")
        self.plotar_grafico()

    def aprovar(self):
        print(
            f"Padrão APROVADO: {self.tipo_padrao} em {self.data_inicio.date()}")
        self.lista_aprovados.append(
            self.df_candidatos.iloc[self.indice_atual].to_dict())
        self.indice_atual += 1
        self.carregar_proximo_padrao()

    def rejeitar(self):
        print(
            f"Padrão REJEITADO: {self.tipo_padrao} em {self.data_inicio.date()}")
        self.indice_atual += 1
        self.carregar_proximo_padrao()

    def salvar_e_sair(self):
        if self.lista_aprovados:
            df_aprovados = pd.DataFrame(self.lista_aprovados)
            df_aprovados.to_csv(self.arquivo_saida, index=False)
            print(
                f"Arquivo de padrões verificados salvo como '{self.arquivo_saida}'!")
        self.root.destroy()


if __name__ == '__main__':
    ARQUIVO_CANDIDATOS = 'todos_os_padroes_btc_automatico.csv'
    ARQUIVO_SAIDA_VERIFICADO = 'dataset_verificado.csv'

    root = tk.Tk()
    app = AnotadorApp(root, ARQUIVO_CANDIDATOS, ARQUIVO_SAIDA_VERIFICADO)
    root.mainloop()
