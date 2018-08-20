import numpy as np
import plotly
from plotly.graph_objs import *

#plotly  範例
def main():

    '''
    trace0 = Scatter(
        x=[1, 2, 3, 4],
        y=[10, 15, 13, 17]
    )
    trace1 = Scatter(
        x=[1, 2, 3, 4],
        y=[16, 5, 11, 9]
    )
    data = Data([trace0, trace1])
    '''

    '''
    #散佈圖
    N = 1000
    random_x = np.random.randn(N)
    random_y = np.random.randn(N)

    # Create a trace
    trace = Scatter(
        x=random_x,
        y=random_y,
        mode='markers'
    )

    data = [trace]
    plotly.offline.plot(data)
    '''


    #散佈圖(兩種顏色)
    N = 500

    trace0 = Scatter(
        x=np.random.randn(N),
        y=np.random.randn(N) + 2,
        name='Above',
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(152, 0, 0, .8)',
            line=dict(
                width=2,
                color='rgb(0, 0, 0)'
            )
        )
    )

    trace1 = Scatter(
        x=np.random.randn(N),
        y=np.random.randn(N) - 2,
        name='Below',
        mode='markers',
        marker=dict(
            size=10,
            color='rgba(255, 182, 193, .9)',
            line=dict(
                width=2,
            )
        )
    )

    data = [trace0, trace1]

    layout = dict(title='Styled Scatter',
                  yaxis=dict(zeroline=False),
                  xaxis=dict(zeroline=False)
                  )

    fig = dict(data=data, layout=layout)
    plotly.offline.plot(fig)

if __name__ == "__main__":
    main()
