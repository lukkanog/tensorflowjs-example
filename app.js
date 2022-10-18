const pontosEixoX = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100];

const valoresEixoY = [0, 5, 10, 15, 20];
const pontosEixoXJaPreenchidos = pontosEixoX.slice(0, valoresEixoY.length)
const pontosEixoXASeremPreenchidos = pontosEixoX.slice(valoresEixoY.length, pontosEixoX.length + 1)

let valoresPrevistos = []

const quantidadeInput = document.getElementById('number')
const botaoSubmit = document.getElementById('submit')

const opcoesGrafico = {
  xAxis: {
    type: 'category',
    data: pontosEixoX,
  },
  yAxis: {
    type: 'value'
  },
  series: [
    {
      name: 'valoresIniciais',
      data: valoresEixoY,
      type: 'line',
      label: {
        show: true
      }
    },
    {
      name: 'valoresPrevistos',
      data: valoresPrevistos,
      type: 'line',
      lineStyle: { color: 'orange' },
      label: {
        show: true
      }
    },
  ],
  tooltip: {
    show: true,
    formatter: `<b>Eixo X:</b> {b0}<br /><b>Eixo Y:</b> {c0}`
  },
  axisPointer: {
    show: true
  }
}

const divGrafico = document.getElementById('chart')

const grafico = echarts.init(divGrafico)
grafico.setOption(opcoesGrafico)

const loading = document.getElementById('loading-img')


function limparResultados() {
  valoresPrevistos.length = 0

  pontosEixoXJaPreenchidos.forEach(pontoX => {
    if (pontoX === pontosEixoXJaPreenchidos[pontosEixoXJaPreenchidos.length - 1]) {
      valoresPrevistos.push(valoresEixoY[pontosEixoXJaPreenchidos.length - 1])
      return
    }
    valoresPrevistos.push(null)
  })
}

async function learnLinear() {
  limparResultados()

  // Criando um modelo de regressão linear
  const model = tf.sequential()

  // Adicionando uma camada densa com 1 neurônio
  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
  }))

  // Configurando o otimizador 
  const learningRate = 0.0001;
  const optimizer = tf.train.sgd(learningRate);
  
  // Compilando o modelo com o otimizador e a função de perda
  model.compile({
    loss: 'meanSquaredError',
    optimizer: optimizer
  })

  // Preparando os dados de treinamento
  const xs = tf.tensor2d(pontosEixoXJaPreenchidos, [pontosEixoXJaPreenchidos.length, 1])
  const ys = tf.tensor2d(valoresEixoY, [valoresEixoY.length, 1])

  // Treinando o modelo
  await model.fit(xs, ys, { epochs: quantidadeInput.value })

  console.log(`==== | Epochs: ${quantidadeInput.value} | ====`)

  // Fazendo as previsões
  for (const valor of pontosEixoXASeremPreenchidos) {
    const valorPrevisto = model.predict(tf.tensor2d([valor], [1, 1]))

    await valorPrevisto.data()
      .then(data => {
        valoresPrevistos.push(data[0])
        console.log(`Eixo X: ${valor} | Eixo Y: ${data[0]}`)
      })
  }
  
  
  console.log(`==================`) 
  console.log(``)

  // Recarrega o gráfico
  grafico.setOption(opcoesGrafico)
}

botaoSubmit.addEventListener('click', () => {
  loading.style.display = 'block'
  learnLinear()
    .then(() => {
      loading.style.display = 'none'
    })
})