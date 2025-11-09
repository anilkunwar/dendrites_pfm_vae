import meshio
import pyvista as pv

# ===== 1. 读取 Exodus 文件 =====
mesh = meshio.read(r"C:\Users\u0178651\Desktop\SimulationResultsMoose\LiSinglev28_GOOD\LiSingle.e")

# ===== 2. 打印信息 =====
print("点数量:", len(mesh.points))
print("单元类型:", mesh.cells[0].type if len(mesh.cells) > 0 else "N/A")
print("point_data:", mesh.point_data.keys())

# ===== 3. 提取二维单元（triangle 或 quad）=====
cells = mesh.cells[0]  # 默认取第一个单元块
cell_type = cells.type
connectivity = cells.data

# 将 meshio 单元转为 PyVista 可用格式
# （PyVista 需要“每个单元节点数 + 节点索引”的展平数组）
import numpy as np
n = connectivity.shape[1]
celltypes = np.full(connectivity.shape[0], pv.CellType.TRIANGLE if n == 3 else pv.CellType.QUAD)
cells_pv = np.hstack([np.full((connectivity.shape[0], 1), n), connectivity]).astype(np.int64).ravel()

# ===== 4. 构建 surface 网格 =====
grid = pv.UnstructuredGrid(cells_pv, celltypes, mesh.points)

# ===== 5. 添加字段 =====
grid.point_data["Phase"] = mesh.point_data["eta"]
grid.point_data["Concentration"] = mesh.point_data["c"]
grid.point_data["Potential"] = mesh.point_data["pot"]

# ===== 6. 绘制三联图（Surface + colormap）=====
plotter = pv.Plotter(shape=(1, 3), window_size=(1800, 600))

plotter.subplot(0, 0)
plotter.add_text("Order Parameter (η)", font_size=12)
plotter.add_mesh(grid, scalars="Phase", cmap="gray", show_edges=False, show_scalar_bar=True)

plotter.subplot(0, 1)
plotter.add_text("Li⁺ Concentration (c)", font_size=12)
plotter.add_mesh(grid, scalars="Concentration", cmap="viridis", show_edges=False, show_scalar_bar=True)

plotter.subplot(0, 2)
plotter.add_text("Electric Potential (φₑ)", font_size=12)
plotter.add_mesh(grid, scalars="Potential", cmap="plasma", show_edges=False, show_scalar_bar=True)

plotter.link_views()
plotter.show()
