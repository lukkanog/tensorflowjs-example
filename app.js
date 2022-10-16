const pontosEixoX = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

const valoresEixoY = [0, 5, 10, 15, 20, 25]

const pontosEixoXJaPreenchidos = pontosEixoX.slice(0, valoresEixoY.length)
const pontosEixoXASeremPreenchidos = pontosEixoX.slice(valoresEixoY.length, pontosEixoX.length + 1)

let valoresPrevistos = []

const quantidadeInput = document.getElementById('number').value
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

  const model = tf.sequential()

  model.add(tf.layers.dense({
    units: 1,
    inputShape: [1]
  }))

  const learningRate = 0.0001;
  const optimizer = tf.train.sgd(learningRate);

  model.compile({
    loss: 'meanSquaredError',
    optimizer: optimizer
  })

  const xs = tf.tensor2d(pontosEixoXJaPreenchidos, [pontosEixoXJaPreenchidos.length, 1])
  const ys = tf.tensor2d(valoresEixoY, [valoresEixoY.length, 1])

  // Vai passar pela rede neural a quantidade de epochs definido abaixo:
  await model.fit(xs, ys, { epochs: quantidadeInput })

  for (const valor of pontosEixoXASeremPreenchidos) {
    const valorPrevisto = model.predict(tf.tensor2d([valor], [1, 1]))

    await valorPrevisto.data()
      .then(data => {
        valoresPrevistos.push(data[0])
        console.log(`Eixo X: ${valor} | Eixo Y: ${data[0]}`)
      })
  }

  //recarrega o gráfico
  grafico.setOption(opcoesGrafico)
}

botaoSubmit.addEventListener('click', () => learnLinear())