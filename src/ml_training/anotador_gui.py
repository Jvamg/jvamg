import tkinter as tk
from tkinter import messagebox
import pandas as pd
import yfinance as yf
import mplfinance as mpf
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import re # Usaremos para interpretar o intervalo

class AnotadorApp:
    def __init__(self, root, arquivo_candidatos, arquivo_saida):
        self.root = root
        self.root.title("Ferramenta de Anotação Multi-Ativo (v3.0)")
        self.root.geometry("1200x850")

        self.arquivo_saida = arquivo_saida

        print("Carregando padrões candidatos do dataset agregado...")
        try:
            # MUDANÇA: As datas já são carregadas do arquivo agregado
            self.df_candidatos = pd.read_csv(arquivo_candidatos, parse_dates=[
                                             'data_inicio', 'data_fim', 'data_cabeca'])
        except FileNotFoundError:
            messagebox.showerror(
                "Erro", f"Arquivo de candidatos '{arquivo_candidatos}' não encontrado! Certifique-se de que o script de geração em lote foi executado.")
            self.root.destroy()
            return
        except Exception as e:
            messagebox.showerror("Erro ao ler CSV", str(e))
            self.root.destroy()
            return

        self.indice_atual = 0
        self.lista_aprovados = []

        # NOVO: Atributos para guardar o contexto do padrão atual
        self.ticker_atual = None
        self.intervalo_atual = None
        
        # --- Layout da Interface (sem grandes mudanças) ---
        self.frame_grafico = tk.Frame(root)
        self.frame_grafico.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self.frame_inferior = tk.Frame(root)
        self.frame_inferior.pack(fill=tk.X, padx=10, pady=5)

        self.info_label = tk.Label(
            self.frame_inferior, text="Carregando...", font=("Arial", 11), justify=tk.LEFT)
        self.info_label.pack(side=tk.LEFT, padx=10, pady=5)

        self.frame_botoes_acao = tk.Frame(self.frame_inferior)
        self.frame_botoes_acao.pack(side=tk.RIGHT, padx=5)

        self.rejeitar_btn = tk.Button(
            self.frame_botoes_acao, text="❌ Rejeitar", command=self.rejeitar, font=("Arial", 12), bg="salmon")
        self.rejeitar_btn.pack(side=tk.RIGHT, padx=5)

        self.aprovar_btn = tk.Button(self.frame_botoes_acao, text="✅ Aprovar",
                                     command=self.aprovar, font=("Arial", 12), bg="lightgreen")
        self.aprovar_btn.pack(side=tk.RIGHT)

        self.frame_ajustes = tk.Frame(self.frame_inferior)
        self.frame_ajustes.pack(side=tk.RIGHT, padx=10)

        tk.Label(self.frame_ajustes, text="Início:", font=("Arial", 10)).grid(row=0, column=0, columnspan=2)
        tk.Button(self.frame_ajustes, text="<<", command=lambda: self.ajustar_data('inicio', 'tras')).grid(row=1, column=0)
        tk.Button(self.frame_ajustes, text=">>", command=lambda: self.ajustar_data('inicio', 'frente')).grid(row=1, column=1)

        tk.Label(self.frame_ajustes, text="Fim:", font=("Arial", 10)).grid(row=0, column=2, columnspan=2, padx=(10, 0))
        tk.Button(self.frame_ajustes, text="<<", command=lambda: self.ajustar_data('fim', 'tras')).grid(row=1, column=2, padx=(10, 0))
        tk.Button(self.frame_ajustes, text=">>", command=lambda: self.ajustar_data('fim', 'frente')).grid(row=1, column=3)

        tk.Label(self.frame_ajustes, text="Cabeça:", font=("Arial", 10)).grid(row=0, column=4, columnspan=2, padx=(10, 0))
        tk.Button(self.frame_ajustes, text="<<", command=lambda: self.ajustar_data('cabeca', 'tras')).grid(row=1, column=4, padx=(10, 0))
        tk.Button(self.frame_ajustes, text=">>", command=lambda: self.ajustar_data('cabeca', 'frente')).grid(row=1, column=5)

        self.canvas = None
        self.carregar_proximo_padrao()

    # MUDANÇA CRÍTICA: O ajuste de data agora é dinâmico com base no intervalo
    def ajustar_data(self, qual_data, direcao):
        if not self.intervalo_atual: return # Segurança
        
        # Lógica para converter o string do intervalo (ex: '4h', '1d') para um Timedelta
        match = re.match(r"(\d+)([hHdD])", self.intervalo_atual)
        if match:
            valor, unidade = match.groups()
            unidade = unidade.lower()
            if unidade == 'h':
                ajuste = pd.Timedelta(hours=int(valor))
            elif unidade in ['d']:
                ajuste = pd.Timedelta(days=int(valor))
            else:
                ajuste = pd.Timedelta(hours=4) # Fallback seguro
        else:
            ajuste = pd.Timedelta(hours=4) # Fallback seguro

        fator = -1 if direcao == 'tras' else 1
        
        if qual_data == 'inicio':
            self.data_inicio_ajustada += (ajuste * fator)
        elif qual_data == 'fim':
            self.data_fim_ajustada += (ajuste * fator)
        elif qual_data == 'cabeca' and self.data_cabeca_ajustada:
            self.data_cabeca_ajustada += (ajuste * fator)

        self.atualizar_info_label()
        self.plotar_grafico()

    # MUDANÇA: plotar_grafico agora com um bloco robusto de limpeza de dados
    def plotar_grafico(self):
        if self.canvas:
            self.canvas.get_tk_widget().destroy()

        data_inicio_contexto = self.data_inicio_ajustada - pd.Timedelta(days=15)
        data_fim_contexto = self.data_fim_ajustada + pd.Timedelta(days=15)

        print(f"Buscando dados para {self.ticker_atual} ({self.intervalo_atual}) de {data_inicio_contexto.date()} a {data_fim_contexto.date()}...")
        
        df_grafico = yf.download(
            tickers=self.ticker_atual, 
            start=data_inicio_contexto, 
            end=data_fim_contexto,
            interval=self.intervalo_atual, 
            progress=False, 
            auto_adjust=True)
        
        if df_grafico.empty:
            print("Não foi possível buscar dados da API. Pulando...")
            self.root.after(100, self.rejeitar)
            return

        if isinstance(df_grafico.columns, pd.MultiIndex):
            df_grafico.columns = df_grafico.columns.get_level_values(0)

        df_grafico.index = pd.to_datetime(df_grafico.index).tz_localize(None)

        print("Sanitizando e validando os tipos de dados...")
        cols_to_validate = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        for col in cols_to_validate:
            if col in df_grafico.columns:
                df_grafico[col] = pd.to_numeric(df_grafico[col], errors='coerce')
            else:
                print(f"Aviso: A coluna '{col}' não foi encontrada nos dados baixados.")

        df_grafico.dropna(subset=['Open', 'High', 'Low', 'Close'], inplace=True)
        
        if df_grafico.empty:
            print("DataFrame ficou vazio após a limpeza de dados. Pulando...")
            self.root.after(100, self.rejeitar)
            return
        
        print("Plotando gráfico...")
        try:
            fig, axlist = mpf.plot(df_grafico, type='candle',
                                   style='charles', returnfig=True, figsize=(10, 6), 
                                   title=f"Analisar: {self.tipo_padrao} em {self.ticker_atual} ({self.intervalo_atual})")
            ax = axlist[0]
            
            # --- CORREÇÃO DE COMPATIBILIDADE ---
            # Esta lógica para encontrar o índice mais próximo funciona em todas as versões do Pandas.
            # 1. Calcula a diferença absoluta de tempo.
            # 2. .argmin() retorna a POSIÇÃO do menor valor.
            start_pos_idx = (df_grafico.index - self.data_inicio_ajustada).to_series().abs().argmin()
            end_pos_idx = (df_grafico.index - self.data_fim_ajustada).to_series().abs().argmin()
            ax.axvspan(start_pos_idx, end_pos_idx, color='yellow', alpha=0.3)

            if self.tipo_padrao in ['OCO', 'OCOI'] and self.data_cabeca_ajustada:
                head_pos_idx = (df_grafico.index - self.data_cabeca_ajustada).to_series().abs().argmin()
                ax.axvline(x=head_pos_idx, color='dodgerblue',
                           linestyle='--', linewidth=1.2, label='Cabeça (Head)')
                ax.legend()
            # --- FIM DA CORREÇÃO ---

        except Exception as e:
            print(f"Aviso: Não foi possível desenhar marcações. Erro ({type(e).__name__}): {e}")

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_grafico)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.canvas.draw()

    # MUDANÇA: Carrega ticker e intervalo do DataFrame
    def carregar_proximo_padrao(self):
        if self.indice_atual >= len(self.df_candidatos):
            messagebox.showinfo("Fim!", "Todos os padrões foram analisados!")
            self.salvar_e_sair()
            return

        padrao_atual = self.df_candidatos.iloc[self.indice_atual]
        
        # NOVO: Carrega o contexto do padrão atual
        self.ticker_atual = padrao_atual['ticker']
        self.intervalo_atual = padrao_atual['intervalo']
        self.tipo_padrao = padrao_atual['tipo_padrao']
        
        self.data_inicio_original = padrao_atual['data_inicio']
        self.data_fim_original = padrao_atual['data_fim']
        self.data_inicio_ajustada = self.data_inicio_original
        self.data_fim_ajustada = self.data_fim_original

        self.data_cabeca_original = None
        self.data_cabeca_ajustada = None
        if 'data_cabeca' in padrao_atual and pd.notna(padrao_atual['data_cabeca']):
            self.data_cabeca_original = padrao_atual['data_cabeca']
            self.data_cabeca_ajustada = self.data_cabeca_original

        self.atualizar_info_label()
        self.plotar_grafico()
    
    # MUDANÇA: Label de informação agora mostra Ticker e Intervalo
    def atualizar_info_label(self):
        info_text = (f"Padrão {self.indice_atual + 1}/{len(self.df_candidatos)} | Ativo: {self.ticker_atual} ({self.intervalo_atual}) | Tipo: {self.tipo_padrao}\n"
                     f"Início: {self.data_inicio_ajustada.strftime('%Y-%m-%d %H:%M')} | "
                     f"Fim: {self.data_fim_ajustada.strftime('%Y-%m-%d %H:%M')}")
        if self.data_cabeca_ajustada:
            info_text += f"\nCabeça: {self.data_cabeca_ajustada.strftime('%Y-%m-%d %H:%M')}"

        self.info_label.config(text=info_text)

    def aprovar(self):
        print(f"Padrão APROVADO: {self.tipo_padrao} em {self.ticker_atual} ({self.data_inicio_original.date()})")

        padrao_aprovado = self.df_candidatos.iloc[self.indice_atual].to_dict()
        padrao_aprovado['data_inicio'] = self.data_inicio_ajustada
        padrao_aprovado['data_fim'] = self.data_fim_ajustada
        padrao_aprovado['data_cabeca'] = self.data_cabeca_ajustada
        padrao_aprovado['status_verificacao'] = 'aprovado'
        padrao_aprovado['ajustado_manualmente'] = 'Sim' if (self.data_inicio_original != self.data_inicio_ajustada or 
                                                           self.data_fim_original != self.data_fim_ajustada or
                                                           self.data_cabeca_original != self.data_cabeca_ajustada) else 'Nao'

        self.lista_aprovados.append(padrao_aprovado)
        self.indice_atual += 1
        self.carregar_proximo_padrao()

    def rejeitar(self):
        print(f"Padrão REJEITADO: {self.tipo_padrao} em {self.ticker_atual} ({self.data_inicio_original.date()})")
        # Opcional: você pode salvar os rejeitados também se quiser
        # padrao_rejeitado = self.df_candidatos.iloc[self.indice_atual].to_dict()
        # padrao_rejeitado['status_verificacao'] = 'rejeitado'
        # self.lista_aprovados.append(padrao_rejeitado)

        self.indice_atual += 1
        self.carregar_proximo_padrao()

    # A função de salvar permanece robusta e já funciona com as novas colunas
    def salvar_e_sair(self):
        if self.lista_aprovados:
            df_aprovados = pd.DataFrame(self.lista_aprovados)
            
            # Reorganiza as colunas para uma melhor leitura
            cols_ordem = [
                'ticker', 'intervalo', 'tipo_padrao', 'data_inicio', 'data_fim', 'data_cabeca',
                'status_verificacao', 'ajustado_manualmente'
            ]
            # Adiciona outras colunas que possam existir (como 'preco_cabeca')
            outras_colunas = [col for col in df_aprovados.columns if col not in cols_ordem]
            df_aprovados = df_aprovados[cols_ordem + outras_colunas]

            df_aprovados.to_csv(self.arquivo_saida, index=False,
                                float_format='%.2f',
                                date_format='%Y-%m-%d %H:%M:%S')
            print(f"Arquivo de padrões verificados salvo como '{self.arquivo_saida}'!")
        self.root.destroy()


if __name__ == '__main__':
    # MUDANÇA: Apontamos para o arquivo agregado como nossa fonte de dados
    ARQUIVO_CANDIDATOS = 'dataset_agregado_hns.csv' 
    ARQUIVO_SAIDA_VERIFICADO = 'dataset_verificado_final.csv'

    root = tk.Tk()
    app = AnotadorApp(root, ARQUIVO_CANDIDATOS, ARQUIVO_SAIDA_VERIFICADO)
    root.mainloop()