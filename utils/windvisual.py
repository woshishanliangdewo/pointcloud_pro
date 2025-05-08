import numpy as np
import pyvista as pv
from data.wind_field_generate import generate_3d_wind_field


# PyVista交互式可视化（支持旋转、缩放）
def pyvista_3d_flow(x, y, z, u, v, w):
    grid = pv.StructuredGrid(x, y, z)
    grid["velocity"] = np.vstack((u.ravel(), v.ravel(), w.ravel())).T
    
    # 创建绘制器
    plotter = pv.Plotter()
    
    # 添加流线
    stream, src = grid.streamlines(
        return_source=True,
        max_time=100,
        terminal_speed=0.01,
        n_points=50,
        source_radius=5
    )
    
    plotter.add_mesh(
        stream.tube(radius=0.05),
        scalars="velocity",
        lighting=True,
        specular=0.5,
        cmap='turbo'
    )
    
    # 添加切片
    slices = grid.slice_orthogonal()
    plotter.add_mesh(slices, opacity=0.5, cmap='jet')
    
    # 添加坐标轴和标题
    plotter.add_axes(xlabel='X', ylabel='Y', zlabel='Z')
    plotter.add_title("3D Wind Field Visualization")
    
    # 添加交互控件
    plotter.add_slider_widget(
        lambda value: update_opacity(value, slices),
        [0.1, 1.0], title='Opacity Control'
    )
    
    plotter.show()


def update_opacity(value, mesh):
    mesh.opacity = value

if __name__ == "__main__":
    # 生成示例数据
    x, y, z, u, v, w = generate_3d_wind_field()
    
    # 选择可视化方式
    visualization_type = 2  # 1=Mayavi, 2=PyVista 

    pyvista_3d_flow(x, y, z, u, v, w)


