import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.cm as cm
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Circle, Wedge, FancyBboxPatch, Rectangle, Polygon
from matplotlib.collections import PatchCollection, LineCollection
import matplotlib.colors as mcolors
import matplotlib.transforms as transforms
from matplotlib.patheffects import withStroke, Normal
import matplotlib.ticker as ticker
import io
import warnings
import gc
from typing import List, Dict, Tuple, Optional, Union
from scipy import stats
from scipy.spatial.distance import pdist, squareform
import colorsys

warnings.filterwarnings('ignore')

# Set global matplotlib parameters
plt.rcParams['figure.max_open_warning'] = 0
plt.rcParams['agg.path.chunksize'] = 10000
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

# ============================================================
# VISUAL DEFAULTS AND CONFIGURATION
# ============================================================

# Global visual defaults with enhanced options
VISUAL_DEFAULTS = {
    # Node styling
    'node_edge_width': 2.0,
    'node_alpha': 0.9,
    'node_edge_color': 'white',
    'node_shadow': False,
    'node_shadow_offset': 0.05,
    'node_shadow_alpha': 0.3,
    'node_shadow_color': 'black',
    'node_glow': False,
    'node_glow_width': 10,
    'node_glow_alpha': 0.2,
    'node_glow_color': 'white',
    'node_gradient': True,
    'node_gradient_direction': 'radial',
    
    # Label styling
    'label_fontsize': 12,
    'label_padding': 0.35,
    'label_offset_radius': 0.35,
    'label_bbox_pad': 0.25,
    'label_bbox_alpha': 0.85,
    'label_bbox_fc': 'white',
    'label_bbox_ec': 'lightgray',
    'label_bbox_lw': 1.0,
    'label_bbox_rounding': 4,
    'label_halo': True,
    'label_halo_width': 3,
    'label_halo_color': 'white',
    'label_rotation': 'auto',
    'label_color': 'black',
    'label_weight': 'bold',
    'label_show_values': True,
    'label_value_format': '.2f',
    'label_value_fontsize_factor': 0.8,
    
    # Edge/connection styling
    'edge_alpha': 0.7,
    'edge_width_scale': 3.0,
    'edge_min_width': 0.5,
    'edge_max_width': 8.0,
    'edge_curvature': 0.35,
    'edge_smoothness': 50,
    'edge_shadow': False,
    'edge_shadow_offset': 0.02,
    'edge_shadow_alpha': 0.2,
    'edge_shadow_color': 'black',
    'edge_glow': False,
    'edge_glow_width': 5,
    'edge_glow_alpha': 0.15,
    'edge_glow_color': 'white',
    'edge_gradient': True,
    'edge_dash_pattern': None,  # 'solid', 'dashed', 'dotted', 'dashdot'
    
    # Radial layout
    'radial_base': 1.0,
    'radial_spacing': 0.8,
    'radial_label_offset': 0.35,
    'radial_grid': True,
    'radial_grid_alpha': 0.1,
    'radial_grid_color': 'gray',
    'radial_grid_style': '--',
    'radial_grid_width': 0.5,
    'radial_ticks': True,
    'radial_tick_count': 5,
    'radial_tick_labels': True,
    'radial_tick_fontsize': 8,
    'circumferential_ticks': True,
    'circumferential_tick_count': 12,
    'circumferential_tick_labels': True,
    'circumferential_tick_fontsize': 8,
    
    # Title and text
    'title_fontsize': 16,
    'title_pad': 30,
    'title_weight': 'bold',
    'title_color': 'black',
    'title_background': False,
    'title_background_color': 'white',
    'title_background_alpha': 0.8,
    'title_halo': False,
    'title_halo_width': 2,
    'title_halo_color': 'white',
    
    # Colorbar styling
    'colorbar_labelsize': 12,
    'colorbar_ticksize': 10,
    'colorbar_pad': 0.12,
    'colorbar_aspect': 50,
    'colorbar_shrink': 1.0,
    'colorbar_extend': 'neither',
    'colorbar_orientation': 'horizontal',
    'colorbar_location': 'bottom',
    'colorbar_label': None,
    'colorbar_ticks': 'auto',
    'colorbar_tick_count': 5,
    
    # Subplot and layout
    'subplot_pad': 0.12,
    'tight_layout': True,
    'constrained_layout': False,
    'figure_dpi': 100,
    'background_color': 'white',
    'background_gradient': False,
    'background_gradient_colors': ['#ffffff', '#f0f0f0'],
    'frame_visible': False,
    'frame_color': 'black',
    'frame_width': 1.0,
    'border_padding': 0.05,
    
    # Advanced visual effects
    'gradient_nodes': True,
    'gradient_edges': True,
    'bezier_control': 'midpoint',
    'anti_aliasing': True,
    'label_collision_avoidance': True,
    'label_collision_buffer': 0.15,
    'auto_rotation_threshold': 90,
    'blend_mode': 'alpha',  # 'alpha', 'multiply', 'screen', 'overlay'
    'light_source': None,  # 'top', 'bottom', 'left', 'right'
    'perspective': False,
    'perspective_strength': 0.1,
    
    # Radar/Radial plot specific
    'radar_fill': False,
    'radar_fill_alpha': 0.2,
    'radar_line_width': 2.0,
    'radar_marker_size': 30,
    'radar_marker_style': 'o',
    'radar_scale': 'linear',  # 'linear', 'log', 'symlog'
    'radar_scale_factor': 1.0,
    'radar_grid_type': 'polygon',  # 'polygon', 'circle', 'spiderweb'
    'radar_grid_levels': 3,
    'radar_axis_labels': True,
    'radar_axis_label_offset': 0.1,
}

# Example data
EXAMPLE_CSV = """step,score,coverage,hopping_strength,t,POT_LEFT,fo,Al,Bl,Cl,As,Bs,Cs,cleq,cseq,L1o,L2o,ko,Noise
0,5.169830808799158,0.10416666666666667,0.1,0.028256090357899666,0.4339944124221802,0.29827773571014404,0.5466359853744507,0.49647387862205505,0.4251793622970581,0.5390684008598328,0.33983609080314636,0.5945582389831543,0.46486878395080566,0.4681702256202698,0.359584778547287,0.6078763008117676,0.4093511402606964,0.31751325726509094
1,5.179830808799158,0.10460069444444445,0.11139433523068368,0.0594908632338047,0.4270627200603485,0.27864405512809753,0.5769832134246826,0.5266308188438416,0.36398613452911377,0.6301330327987671,0.35744181275367737,0.5121908187866211,0.3641984462738037,0.4810342490673065,0.37694957852363586,0.612697958946228,0.4058190882205963,0.34840521216392517
2,5.304021017662722,0.10460069444444445,0.12041199826559248,0.08961087465286255,0.4117993414402008,0.3086051940917969,0.5759025812149048,0.5457133650779724,0.37897956371307373,0.6828550696372986,0.37128087878227234,0.5426892042160034,0.31361812353134155,0.5300193428993225,0.3330453932285309,0.6396515965461731,0.4162127375602722,0.386873722076416
3,5.169830808799158,0.11197916666666667,0.1278753600952829,0.11320602893829346,0.5374900102615356,0.3893888592720032,0.6208136677742004,0.5366111993789673,0.4114663004875183,0.7501887679100037,0.4020015299320221,0.45790043473243713,0.343185693025589,0.7146779298782349,0.3623429834842682,0.6474707126617432,0.4218909740447998,0.414340615272522
4,5.169830808799158,0.11371527777777778,0.13424226808222062,0.1594301164150238,0.5966717004776001,0.4405992031097412,0.593701183795929,0.6432862877845764,0.48903852701187134,0.8087910413742065,0.4361550211906433,0.5463976263999939,0.351766973733902,0.7200356721878052,0.39372894167900085,0.699080228805542,0.46395444869995117,0.33751198649406433
5,5.1838690114837185,0.12890625,0.13979400086720378,0.27053752541542053,0.5414740443229675,0.3837757110595703,0.5459027290344238,0.7114285826683044,0.5601555109024048,0.8636823296546936,0.4749370515346527,0.5336172580718994,0.49408119916915894,0.6982162594795227,0.4363292157649994,0.6871276497840881,0.5171192288398743,0.34152668714523315
6,5.100333034308876,0.1440972222222222,0.14471580313422192,0.3258400559425354,0.6228137016296387,0.49862906336784363,0.6784278154373169,0.8143172264099121,0.5625902414321899,0.896495521068573,0.6146658062934875,0.5892767310142517,0.4974607229232788,0.8461188077926636,0.5145464539527893,0.8220916986465454,0.6068583726882935,0.40622031688690186
7,7.32416026628129,0.1840277777777778,0.14913616938342728,0.3501424491405487,0.3400475084781647,0.5205002427101135,0.697181224822998,0.9804031848907471,0.6573023200035095,0.8734657168388367,0.8090043663978577,0.7597450017929077,0.57285475730896,0.7757821083068848,0.645453929901123,0.8650208711624146,0.5720763802528381,0.4552328288555145
8,10.386888918380697,0.22178819444444445,0.15314789170422552,0.377905011177063,0.3762165307998657,0.5274438858032227,0.6071285009384155,0.9385737180709839,0.6077378988265991,0.8730340600013733,0.9395252466201782,0.7045806646347046,0.5667017102241516,0.7497444152832031,0.8033575415611267,0.7816545963287354,0.5264078378677368,0.5300353765487671
9,8.254108578194927,0.3077256944444444,0.1568201724066995,0.3872142434120178,0.47770050168037415,0.4989684224128723,0.7149577140808105,1.0060926675796509,0.6737095713615417,0.7603132724761963,0.6836521625518799,0.6091089844703674,0.4301818609237671,0.8038907051086426,0.7902536988258362,0.9215232133865356,0.5898656249046326,0.6257796287536621
10,5.622075167395693,0.3493923611111111,0.16020599913279623,0.3937322199344635,0.33547544479370117,0.44358500838279724,0.7463698983192444,0.8410062193870544,0.6607444286346436,0.6670836806297302,0.6114344596862793,0.6689836382865906,0.4694424569606781,0.7656615376472473,0.7240639328956604,0.8651359677314758,0.7883168458938599,0.6361547112464905
11,7.321904299308076,0.3650173611111111,0.16334684555795864,0.45904433727264404,0.27944284677505493,0.38179677724838257,0.8159457445144653,1.0150943994522095,0.7721518874168396,0.7039114236831665,0.7788766026496887,0.7146820425987244,0.49762389063835144,0.7908686399459839,0.8121711611747742,0.8742984533309937,0.7975523471832275,0.6416141390800476
12,8.543683990691765,0.3723958333333333,0.1662757831681574,0.5105248689651489,0.34250542521476746,0.43922367691993713,0.7341154217720032,1.1102296113967896,0.7483778595924377,0.7128080725669861,0.7358441352844238,0.7393836975097656,0.46837639808654785,0.8202276825904846,0.877817690372467,0.9840682744979858,0.7861328125,0.6163419485092163
13,7.620555709528761,0.3732638888888889,0.16901960800285137,0.5417518019676208,0.43418756127357483,0.40142834186553955,0.7275840044021606,1.0893441438674927,0.7854703664779663,0.7538084983825684,0.6650506258010864,0.796431303024292,0.4985898435115814,0.8372722864151001,0.744591236114502,1.123425841331482,0.785809338092804,0.680273711681366
14,6.3356596618964565,0.3745659722222222,0.17160033436347993,0.5423417687416077,0.26539722084999084,0.5920475721359253,0.6683412790298462,1.082399845123291,0.6814282536506653,0.8131998181343079,0.6773099303245544,0.7042008638381958,0.5497509837150574,0.6867591738700867,0.8954933881759644,0.8905871510505676,0.7955514192581177,0.6689854264259338
15,7.436352609042917,0.40625,0.17403626894942437,0.568615198135376,0.24044141173362732,0.6975681781768799,0.6411821246147156,1.1813530921936035,0.7024039626121521,0.8560113310813904,0.709820568561554,0.76691734790802,0.6928378343582153,0.4951046407222748,0.9582697153091431,0.9249935150146484,0.7339156866073608,0.7971505522727966
16,10.277262759911848,0.4618055555555556,0.17634279935629374,0.5763835906982422,0.46326711773872375,0.8927175998687744,0.6113301515579224,1.1488991975784302,0.8268088698387146,0.8452023863792419,0.6038599610328674,0.7312695980072021,0.7745003700256348,0.523252010345459,0.9515889286994934,0.7617896795272827,0.7811905741691589,0.7240396738052368
17,10.171294809365978,0.46484375,0.17853298350107671,0.6809343695640564,0.5394753217697144,0.7823241353034973,0.5580039620399475,1.2827746868133545,0.8729355335235596,0.9322726130485535,0.4856688380241394,0.942378580570221,0.7398176789283752,0.6016049981117249,0.8035934567451477,0.7253121137619019,0.6769372820854187,0.6313362121582031
18,10.629263728382883,0.48046875,0.1806179973983887,0.6865774393081665,0.5340346097946167,0.8711283802986145,0.6346777677536011,1.232739806175232,0.9320783019065857,0.9418307542800903,0.49510204792022705,1.0256969928741455,0.7569521069526672,0.6322088837623596,0.8393365144729614,0.7755464911460876,0.706352949142456,0.6519243717193604
19,10.49475562025519,0.4822048611111111,0.18260748027008264,0.7152098417282104,0.4992449879646301,0.9342008233070374,0.8139623403549194,1.3508509397506714,1.0962871313095093,0.9878286719322205,0.6010639071464539,0.9243364930152893,0.9159383177757263,0.6504509449005127,0.913641095161438,0.8966460227966309,0.9222412705421448,0.878772497177124
20,10.22933457672259,0.4887152777777778,0.1845098040014257,0.7465022206306458,0.5899507403373718,0.9446436762809753,0.8453498482704163,1.564751148223877,1.0583618879318237,1.1709020137786865,0.7551482319831848,0.9316735863685608,1.0114786624908447,0.7386069893836975,0.8721351027488708,0.712131679058075,0.9934794902801514,0.8559936881065369
21,10.40191081814774,0.5212673611111112,0.1863322860120456,0.778833270072937,0.46530479192733765,0.9134830236434937,0.8916224837303162,1.5924957990646362,1.0132859945297241,1.3060201406478882,0.7646929621696472,1.0435600280761719,0.8411058783531189,0.7712708711624146,0.9179795384407043,0.8546093702316284,0.9429539442062378,0.8560317158699036
22,11.174144433501233,0.5386284722222222,0.18808135922807911,0.8746334314346313,0.48324069380760193,0.9938526749610901,0.769467830657959,1.6170459985733032,1.01220703125,1.2132935523986816,0.752862811088562,1.16727614402771,0.9052226543426514,0.6134433150291443,0.9348587393760681,0.8925090432167053,0.8277872204780579,0.9739061594009399
23,10.844935844038055,0.5503472222222222,0.18976270912904414,0.9238934516906738,0.5506340861320496,0.9444273114204407,0.8249781131744385,1.7516707181930542,0.8675052523612976,1.4430488348007202,0.8421710729598999,1.072837471961975,1.066247820854187,0.9110193252563477,0.8453015089035034,0.6934573650360107,0.8824711441993713,0.9735444188117981
24,9.859185254537385,0.5525173611111112,0.19138138523837167,0.9569121599197388,0.6011497378349304,1.018949031829834,0.8612451553344727,1.6675461530685425,0.9851047396659851,1.3370230197906494,0.9144415855407715,1.1276476383209229,1.0655651092529297,0.8532272577285767,0.8896656036376953,0.7149348855018616,0.8930221796035767,1.089601993560791
25,9.733399429950971,0.5572916666666666,0.19294189257142927,0.9699379205703735,0.5459413528442383,0.9786086082458496,0.9722445607185364,1.618293046951294,1.0138611793518066,1.2777221202850342,0.769503653049469,1.2075700759887695,0.9359092116355896,0.8779188990592957,0.8745647072792053,0.5999792814254761,0.8872678279876709,1.0222569704055786
26,9.9260127892421,0.5690104166666666,0.19444826721501687,1.0167112350463867,0.7404793500900269,1.2331862449645996,0.9591841697692871,1.8226509094238281,1.1032135486602783,1.5929903984069824,1.0649257898330688,1.510267734527588,0.9941714406013489,1.090094804763794,0.8625853061676025,0.5799097418785095,0.9083585143089294,0.8630198836326599
27,10.678004251308256,0.5798611111111112,0.19590413923210936,1.1897140741348267,0.8140103220939636,1.1114305257797241,0.8603397011756897,1.7859822511672974,1.1018062829971313,1.5066109895706177,0.9222522377967834,1.4768295288085938,0.9960513710975647,0.8831607103347778,0.8123284578323364,0.8251258730888367,0.7022106647491455,0.8924230933189392
28,11.020299291797908,0.5798611111111112,0.19731278535996988,1.1938612461090088,0.8728200793266296,1.2563399076461792,0.9425970911979675,1.665657877922058,1.1347849369049072,1.6763732433319092,0.9476985335350037,1.450101613998413,0.9543245434761047,0.7781178951263428,0.7439810633659363,0.7765737175941467,0.9004281163215637,0.8755770325660706
29,11.988932619625839,0.5911458333333334,0.19867717342662447,1.2421698570251465,0.9625719785690308,1.3612264394760132,0.8735606670379639,1.6014152765274048,1.1779128313064575,1.7611147165298462,0.9013581275939941,1.336272954940796,0.8664807081222534,0.7415759563446045,0.8296802639961243,0.8429422974586487,1.0249007940292358,0.7863231301307678
30,11.784495105579659,0.5933159722222222,0.2,1.3608571290969849,0.9625561833381653,1.2631572484970093,0.9952393770217896,1.82819402217865,1.1884130239486694,1.8113062381744385,0.981305718421936,1.5592151880264282,0.9318991899490356,0.6798413395881653,0.864549994468689,0.8141895532608032,0.9140636324882507,0.9269731044769287"""

# Enhanced colormap list with descriptions
COLORMAP_CATEGORIES = {
    'Sequential': ['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'summer', 'autumn', 'winter', 'hot', 'cool', 'copper', 'bone', 'pink', 'gray', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'],
    'Diverging': ['coolwarm', 'RdBu', 'RdGy', 'PiYG', 'PRGn', 'RdYlBu', 'RdYlGn', 'Spectral', 'bwr', 'seismic', 'BrBG', 'PuOr'],
    'Cyclic': ['twilight', 'twilight_shifted', 'hsv'],
    'Qualitative': ['tab10', 'tab20', 'tab20b', 'tab20c', 'Set1', 'Set2', 'Set3', 'Pastel1', 'Pastel2', 'Paired', 'Accent', 'Dark2'],
    'Perceptually Uniform': ['viridis', 'plasma', 'inferno', 'magma', 'cividis'],
    'Miscellaneous': ['jet', 'rainbow', 'turbo', 'gist_rainbow', 'gist_ncar', 'gist_stern', 'nipy_spectral', 'flag', 'prism']
}

ALL_COLORMAPS = []
for category in COLORMAP_CATEGORIES.values():
    ALL_COLORMAPS.extend(category)
ALL_COLORMAPS = sorted(set(ALL_COLORMAPS))

FEATURE_COLS = ['t','POT_LEFT','fo','Al','Bl','Cl','As','Bs','Cs','cleq','cseq','L1o','L2o','ko','Noise']
NODE_NAMES = FEATURE_COLS

# ============================================================
# HELPER FUNCTIONS FOR VISUAL ENHANCEMENTS
# ============================================================

@st.cache_data(ttl=3600, max_entries=10)
def load_data(uploaded_file=None, use_example=True):
    """Load data with caching"""
    if not use_example and uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            return df
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
            return None
    else:
        df = pd.read_csv(io.StringIO(EXAMPLE_CSV))
        return df

def compute_statistics(df_selected):
    """Compute all required statistics efficiently"""
    stats = {}
    
    stats['feature_means'] = df_selected[FEATURE_COLS].mean().values
    stats['feature_stds'] = df_selected[FEATURE_COLS].std().values
    stats['feature_mins'] = df_selected[FEATURE_COLS].min().values
    stats['feature_maxs'] = df_selected[FEATURE_COLS].max().values
    stats['feature_medians'] = df_selected[FEATURE_COLS].median().values
    
    stats['correlation_matrix'] = df_selected[FEATURE_COLS].corr().fillna(0).values
    stats['covariance_matrix'] = df_selected[FEATURE_COLS].cov().fillna(0).values
    
    return stats

def apply_text_halo(text_obj, halo_width=3, halo_color='white'):
    """Apply halo effect to text for better readability"""
    text_obj.set_path_effects([
        withStroke(linewidth=halo_width, foreground=halo_color),
        Normal()
    ])

def create_gradient_color(base_color, factor=0.7, direction='lighter'):
    """Create gradient color from base color"""
    if len(base_color) == 4:
        r, g, b, a = base_color
    else:
        r, g, b = base_color[:3]
        a = 1.0 if len(base_color) == 3 else base_color[3]
    
    if direction == 'lighter':
        return (min(1, r * (1 + factor)), min(1, g * (1 + factor)), min(1, b * (1 + factor)), a)
    else:
        return (r * factor, g * factor, b * factor, a)

def adjust_label_rotation(angle_degrees, auto_rotation_threshold=90):
    """Adjust label rotation for readability"""
    if 90 <= angle_degrees <= 270:
        angle_degrees += 180
        ha = "right"
    else:
        ha = "left"
    
    return angle_degrees, ha

def create_rounded_rectangle(xy, width, height, radius=0.1, **kwargs):
    """Create a rounded rectangle patch"""
    return FancyBboxPatch(xy, width, height,
                         boxstyle=f"round,pad={radius},rounding_size={radius}",
                         **kwargs)

def darken_color(color, factor=0.7):
    """Darken a color by a factor"""
    if len(color) == 4:
        r, g, b, a = color
    else:
        r, g, b = color[:3]
        a = 1.0 if len(color) == 3 else color[3]
    
    return (r * factor, g * factor, b * factor, a)

def lighten_color(color, factor=0.3):
    """Lighten a color by a factor"""
    if len(color) == 4:
        r, g, b, a = color
    else:
        r, g, b = color[:3]
        a = 1.0 if len(color) == 3 else color[3]
    
    return (min(1, r + (1 - r) * factor), min(1, g + (1 - g) * factor), min(1, b + (1 - b) * factor), a)

def get_contrast_color(color):
    """Get contrasting text color (black or white) for given background color"""
    if len(color) == 4:
        r, g, b, _ = color
    else:
        r, g, b = color[:3]
    
    # Calculate luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return 'white' if luminance < 0.5 else 'black'

def create_gradient_cmap(base_color, n_colors=256):
    """Create a gradient colormap from a base color"""
    base_rgb = mcolors.to_rgb(base_color)
    
    # Create dark and light versions
    dark_color = darken_color(base_rgb, 0.3)
    light_color = lighten_color(base_rgb, 0.3)
    
    # Create colormap
    colors = [dark_color, base_rgb, light_color]
    return mcolors.LinearSegmentedColormap.from_list(f"gradient_{base_color}", colors, N=n_colors)

# ============================================================
# ENHANCED CHORD DIAGRAM WITH FULL PARAMETERIZATION
# ============================================================

def create_enhanced_chord_diagram(
    stats,
    metric='correlation',
    colormap_name='coolwarm',
    edge_threshold=0.3,
    max_connections=100,
    figsize=(14, 14),
    title="",
    
    # === NODE STYLING ===
    node_edge_width=2.0,
    node_alpha=0.9,
    node_edge_color='white',
    node_shadow=False,
    node_shadow_offset=0.05,
    node_shadow_alpha=0.3,
    node_shadow_color='black',
    node_glow=False,
    node_glow_width=10,
    node_glow_alpha=0.2,
    node_glow_color='white',
    gradient_nodes=True,
    node_gradient_direction='radial',
    
    # === LABEL STYLING ===
    label_fontsize=12,
    label_padding=0.35,
    label_offset_radius=0.35,
    label_bbox_pad=0.25,
    label_bbox_alpha=0.85,
    label_bbox_fc='white',
    label_bbox_ec='lightgray',
    label_bbox_lw=1.0,
    label_bbox_rounding=4,
    label_halo=True,
    label_halo_width=3,
    label_halo_color='white',
    label_rotation='auto',
    label_color='black',
    label_weight='bold',
    label_show_values=True,
    label_value_format='.2f',
    label_value_fontsize_factor=0.8,
    
    # === EDGE/CONNECTION STYLING ===
    edge_alpha=0.7,
    edge_width_scale=3.0,
    edge_min_width=0.5,
    edge_max_width=8.0,
    edge_curvature=0.35,
    edge_smoothness=50,
    edge_shadow=False,
    edge_shadow_offset=0.02,
    edge_shadow_alpha=0.2,
    edge_shadow_color='black',
    edge_glow=False,
    edge_glow_width=5,
    edge_glow_alpha=0.15,
    edge_glow_color='white',
    gradient_edges=True,
    bezier_control='midpoint',
    edge_dash_pattern=None,
    
    # === RADIAL LAYOUT ===
    radial_base=1.0,
    radial_spacing=0.8,
    radial_label_offset=0.35,
    radial_grid=False,
    radial_grid_alpha=0.1,
    radial_grid_color='gray',
    radial_grid_style='--',
    radial_grid_width=0.5,
    radial_ticks=True,
    radial_tick_count=5,
    radial_tick_labels=True,
    radial_tick_fontsize=8,
    circumferential_ticks=True,
    circumferential_tick_count=12,
    circumferential_tick_labels=True,
    circumferential_tick_fontsize=8,
    
    # === TITLE AND TEXT ===
    title_fontsize=16,
    title_pad=30,
    title_weight='bold',
    title_color='black',
    title_background=False,
    title_background_color='white',
    title_background_alpha=0.8,
    title_halo=False,
    title_halo_width=2,
    title_halo_color='white',
    
    # === COLORBAR STYLING ===
    colorbar_labelsize=12,
    colorbar_ticksize=10,
    colorbar_pad=0.12,
    colorbar_aspect=50,
    colorbar_shrink=1.0,
    colorbar_extend='neither',
    colorbar_orientation='horizontal',
    colorbar_location='bottom',
    colorbar_label=None,
    colorbar_ticks='auto',
    colorbar_tick_count=5,
    
    # === SUBPLOT AND LAYOUT ===
    subplot_pad=0.12,
    tight_layout=True,
    constrained_layout=False,
    figure_dpi=100,
    background_color='white',
    background_gradient=False,
    background_gradient_colors=['#ffffff', '#f0f0f0'],
    frame_visible=False,
    frame_color='black',
    frame_width=1.0,
    border_padding=0.05,
    
    # === ADVANCED VISUAL EFFECTS ===
    anti_aliasing=True,
    label_collision_avoidance=True,
    label_collision_buffer=0.15,
    auto_rotation_threshold=90,
    blend_mode='alpha',
    light_source=None,
    perspective=False,
    perspective_strength=0.1,
):
    """
    Create an enhanced chord diagram with full visual parameterization
    
    Parameters:
    -----------
    All parameters correspond to visual controls listed above
    
    Returns:
    --------
    matplotlib.figure.Figure
        The enhanced chord diagram figure
    """
    
    # -----------------------
    # Matrix selection and setup
    # -----------------------
    if metric == 'correlation':
        matrix = stats['correlation_matrix']
        vmin, vmax = -1, 1
        metric_label = "Correlation"
    else:
        matrix = stats['covariance_matrix']
        max_abs = np.max(np.abs(matrix))
        vmin, vmax = -max_abs, max_abs
        metric_label = "Covariance"
    
    n = len(FEATURE_COLS)
    means = stats['feature_means']
    
    # -----------------------
    # Figure setup with enhanced DPI
    # -----------------------
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=figure_dpi,
        subplot_kw={'projection': 'polar'},
        constrained_layout=constrained_layout,
        facecolor=background_color
    )
    
    # Set background color
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
    # Add background gradient if requested
    if background_gradient:
        gradient = np.linspace(0, 1, 256).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))
        
        # Create gradient colors
        cmap_gradient = mcolors.LinearSegmentedColormap.from_list(
            "bg_gradient", background_gradient_colors, N=256
        )
        
        # Display gradient
        ax.imshow(gradient, aspect='auto', cmap=cmap_gradient,
                 extent=[0, 2*np.pi, 0, radial_base + 2.0],
                 alpha=0.3, transform=ax.transAxes, zorder=-100)
    
    # Polar plot setup
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.set_ylim(0, radial_base + 2.0)
    
    # Configure grid and ticks
    if radial_grid:
        ax.grid(True, alpha=radial_grid_alpha, color=radial_grid_color,
                linestyle=radial_grid_style, linewidth=radial_grid_width)
    else:
        ax.grid(False)
    
    # Configure radial ticks
    if radial_ticks:
        ax.set_yticks(np.linspace(0, radial_base + 1.5, radial_tick_count))
        if radial_tick_labels:
            ax.set_yticklabels([f"{tick:.1f}" for tick in np.linspace(0, radial_base + 1.5, radial_tick_count)],
                              fontsize=radial_tick_fontsize)
        else:
            ax.set_yticklabels([])
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    
    # Configure circumferential ticks
    if circumferential_ticks:
        ax.set_xticks(np.linspace(0, 2*np.pi, circumferential_tick_count, endpoint=False))
        if circumferential_tick_labels:
            angles_deg = np.linspace(0, 360, circumferential_tick_count, endpoint=False)
            ax.set_xticklabels([f"{angle:.0f}°" for angle in angles_deg],
                              fontsize=circumferential_tick_fontsize)
        else:
            ax.set_xticklabels([])
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    
    if not frame_visible:
        ax.spines['polar'].set_visible(False)
    
    # -----------------------
    # Angles and positions
    # -----------------------
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    angles = np.roll(angles, -1)  # Rotate for better label placement
    
    # -----------------------
    # Colormaps
    # -----------------------
    edge_cmap = cm.get_cmap(colormap_name)
    node_cmap = cm.get_cmap("tab20c", n)
    
    # -----------------------
    # Node scaling and colors
    # -----------------------
    if means.max() > 0:
        norm_means = means / means.max()
    else:
        norm_means = np.ones(n)
    
    node_sizes = 0.4 + 0.6 * norm_means
    
    # -----------------------
    # Draw nodes with enhanced effects
    # -----------------------
    node_patches = []
    node_positions = []
    
    for i, angle in enumerate(angles):
        r = radial_base + node_sizes[i] / 2
        node_positions.append((angle, r))
        
        # Base node color
        base_color = node_cmap(i)
        
        # Apply gradient if requested
        if gradient_nodes:
            if node_gradient_direction == 'radial':
                # Create gradient from center to edge
                inner_color = darken_color(base_color, 0.5)
                outer_color = lighten_color(base_color, 0.3)
            else:
                inner_color = darken_color(base_color, 0.3)
                outer_color = lighten_color(base_color, 0.2)
        else:
            inner_color = outer_color = base_color
        
        # Create node with transform for proper orientation
        transform = ax.transData._b + transforms.Affine2D().rotate(angle)
        
        # Add shadow if requested
        if node_shadow:
            shadow = Circle(
                (0, 0),
                r + node_shadow_offset,
                facecolor=node_shadow_color,
                edgecolor='none',
                alpha=node_shadow_alpha,
                transform=transform,
                zorder=1
            )
            ax.add_patch(shadow)
        
        # Add glow effect if requested
        if node_glow:
            glow = Circle(
                (0, 0),
                r + node_glow_width / 100,
                facecolor=node_glow_color,
                edgecolor='none',
                alpha=node_glow_alpha,
                transform=transform,
                zorder=1.5
            )
            ax.add_patch(glow)
        
        # Main node
        node = Circle(
            (0, 0),
            r,
            facecolor=inner_color,
            edgecolor=node_edge_color,
            linewidth=node_edge_width,
            alpha=node_alpha,
            transform=transform,
            zorder=2
        )
        ax.add_patch(node)
        
        # Add gradient effect if requested
        if gradient_nodes and node_gradient_direction == 'radial':
            # Add outer ring for gradient effect
            outer_ring = Circle(
                (0, 0),
                r,
                facecolor='none',
                edgecolor=outer_color,
                linewidth=node_edge_width * 2,
                alpha=node_alpha * 0.5,
                transform=transform,
                zorder=2.1
            )
            ax.add_patch(outer_ring)
        
        node_patches.append(node)
    
    # -----------------------
    # Draw labels with enhanced styling
    # -----------------------
    label_objects = []
    label_positions = []
    
    for i, (angle, r) in enumerate(node_positions):
        # Calculate label position
        label_r = r + label_offset_radius + label_padding
        
        # Determine label rotation
        if label_rotation == 'auto':
            rotation_deg, ha = adjust_label_rotation(np.degrees(angle), auto_rotation_threshold)
        elif label_rotation == 'tangential':
            rotation_deg = np.degrees(angle) + 90
            ha = "center"
        elif label_rotation == 'horizontal':
            rotation_deg = 0
            ha = "center"
        elif label_rotation == 'radial':
            rotation_deg = np.degrees(angle)
            ha = "center"
        else:
            rotation_deg = np.degrees(angle)
            ha = "left" if angle < np.pi else "right"
        
        # Create label text
        label_text = NODE_NAMES[i]
        
        # Add value annotation if requested
        if label_show_values and label_fontsize > 8:
            value_text = format(means[i], label_value_format)
            label_text = f"{NODE_NAMES[i]}\n{value_text}"
        
        # Create text object
        txt = ax.text(
            angle,
            label_r,
            label_text,
            fontsize=label_fontsize,
            fontweight=label_weight,
            color=label_color,
            rotation=rotation_deg,
            rotation_mode="anchor",
            ha=ha,
            va="center",
            zorder=10
        )
        
        # Apply halo effect if requested
        if label_halo:
            apply_text_halo(txt, halo_width=label_halo_width, halo_color=label_halo_color)
        
        # Add bounding box if requested
        if label_bbox_pad > 0:
            txt.set_bbox(dict(
                boxstyle=f"round,pad={label_bbox_pad},rounding_size={label_bbox_rounding}",
                facecolor=label_bbox_fc,
                edgecolor=label_bbox_ec,
                linewidth=label_bbox_lw,
                alpha=label_bbox_alpha
            ))
        
        label_objects.append(txt)
        label_positions.append((angle, label_r))
    
    # -----------------------
    # Handle label collisions
    # -----------------------
    if label_collision_avoidance and len(label_positions) > 1:
        # Simple collision avoidance by adjusting radial position
        for i in range(n):
            for j in range(i + 1, n):
                angle_i, r_i = label_positions[i]
                angle_j, r_j = label_positions[j]
                
                # Calculate angular distance
                angular_dist = min(abs(angle_i - angle_j), 2*np.pi - abs(angle_i - angle_j))
                
                # If labels are too close, adjust radial positions
                if angular_dist < label_collision_buffer:
                    if r_i == r_j:
                        label_positions[i] = (angle_i, r_i + 0.05)
                        label_objects[i].set_position((angle_i, r_i + 0.05))
                    elif abs(r_i - r_j) < 0.1:
                        label_positions[i] = (angle_i, r_i + 0.1)
                        label_objects[i].set_position((angle_i, r_i + 0.1))
    
    # -----------------------
    # Collect and sort connections
    # -----------------------
    connections = []
    for i in range(n):
        for j in range(i + 1, n):
            value = matrix[i, j]
            abs_value = abs(value)
            if abs_value >= edge_threshold:
                connections.append((i, j, value, abs_value))
    
    # Sort by absolute value (strongest first for z-ordering)
    connections.sort(key=lambda x: x[3], reverse=True)
    
    # Limit number of connections for performance
    if len(connections) > max_connections:
        connections = connections[:max_connections]
    
    # -----------------------
    # Draw connections with enhanced styling
    # -----------------------
    for i, j, value, magnitude in connections:
        # Get node positions
        theta1, r1 = node_positions[i]
        theta2, r2 = node_positions[j]
        
        # Calculate intermediate points for Bezier curve
        t = np.linspace(0, 1, edge_smoothness)
        
        # Different bezier control strategies
        if bezier_control == 'midpoint':
            # Midpoint control
            control_angle = (theta1 + theta2) / 2
            control_radius = (r1 + r2) / 2 * (1 - edge_curvature)
            
            # Quadratic Bezier in polar coordinates
            theta = (1-t)**2 * theta1 + 2*(1-t)*t * control_angle + t**2 * theta2
            r = (1-t)**2 * r1 + 2*(1-t)*t * control_radius + t**2 * r2
            
        elif bezier_control == 'adaptive':
            # Adaptive control based on angle difference
            angle_diff = min(abs(theta1 - theta2), 2*np.pi - abs(theta1 - theta2))
            control_factor = 0.7 - 0.3 * (angle_diff / np.pi)
            
            control_angle = (theta1 + theta2) / 2
            control_radius = (r1 + r2) / 2 * control_factor
            
            theta = (1-t)**2 * theta1 + 2*(1-t)*t * control_angle + t**2 * theta2
            r = (1-t)**2 * r1 + 2*(1-t)*t * control_radius + t**2 * r2
            
        else:  # 'fixed'
            # Fixed curvature
            theta = theta1 + (theta2 - theta1) * t
            r = r1 + (r2 - r1) * t
            r = r + np.sin(np.pi * t) * edge_curvature * (r1 + r2) / 2
        
        # Calculate edge color
        color_value = (value - vmin) / (vmax - vmin)
        base_color = edge_cmap(color_value)
        
        # Apply gradient if requested
        if gradient_edges:
            # Create gradient along the edge
            colors = []
            for t_val in t:
                if t_val < 0.5:
                    # Blend from node i color to edge color
                    blend = t_val * 2
                    node_color = node_cmap(i)
                    mixed_color = (
                        node_color[0] * (1 - blend) + base_color[0] * blend,
                        node_color[1] * (1 - blend) + base_color[1] * blend,
                        node_color[2] * (1 - blend) + base_color[2] * blend,
                        edge_alpha
                    )
                else:
                    # Blend from edge color to node j color
                    blend = (t_val - 0.5) * 2
                    node_color = node_cmap(j)
                    mixed_color = (
                        base_color[0] * (1 - blend) + node_color[0] * blend,
                        base_color[1] * (1 - blend) + node_color[1] * blend,
                        base_color[2] * (1 - blend) + node_color[2] * blend,
                        edge_alpha
                    )
                colors.append(mixed_color)
        else:
            color = base_color
        
        # Calculate edge width with limits
        linewidth = max(edge_min_width, min(edge_max_width, magnitude * edge_width_scale))
        
        # Set line style
        linestyle = edge_dash_pattern if edge_dash_pattern else '-'
        
        # Add shadow if requested
        if edge_shadow:
            shadow_theta = theta + edge_shadow_offset
            shadow_r = r + edge_shadow_offset
            ax.plot(
                shadow_theta, shadow_r,
                color=edge_shadow_color,
                linewidth=linewidth,
                alpha=edge_shadow_alpha,
                solid_capstyle='round',
                linestyle=linestyle,
                zorder=3
            )
        
        # Add glow effect if requested
        if edge_glow:
            glow_width = linewidth + edge_glow_width
            ax.plot(
                theta, r,
                color=edge_glow_color,
                linewidth=glow_width,
                alpha=edge_glow_alpha,
                solid_capstyle='round',
                linestyle=linestyle,
                zorder=4
            )
        
        # Main edge
        if gradient_edges:
            # Plot with gradient colors
            for k in range(len(t) - 1):
                segment_theta = theta[k:k+2]
                segment_r = r[k:k+2]
                segment_color = colors[k]
                
                ax.plot(
                    segment_theta, segment_r,
                    color=segment_color,
                    linewidth=linewidth,
                    solid_capstyle='round',
                    linestyle=linestyle,
                    zorder=5
                )
        else:
            # Plot with single color
            ax.plot(
                theta, r,
                color=color,
                linewidth=linewidth,
                alpha=edge_alpha,
                solid_capstyle='round',
                linestyle=linestyle,
                zorder=5
            )
        
        # Add value annotation for strong connections
        if magnitude > 0.8 and label_fontsize > 8:
            mid_idx = len(t) // 2
            label_angle = theta[mid_idx]
            label_radius = r[mid_idx] - 0.1
            
            # Ensure label is readable
            if label_radius > radial_base + 0.5:
                value_text = format(value, label_value_format)
                
                # Determine text color based on background
                if gradient_edges:
                    bg_color = colors[mid_idx]
                else:
                    bg_color = base_color
                
                text_color = get_contrast_color(bg_color)
                
                val_txt = ax.text(
                    label_angle,
                    label_radius,
                    value_text,
                    fontsize=int(label_fontsize * label_value_fontsize_factor),
                    fontweight='bold',
                    ha='center',
                    va='center',
                    color=text_color,
                    zorder=10
                )
                
                # Add halo for better readability
                apply_text_halo(val_txt, halo_width=2, halo_color='black')
    
    # -----------------------
    # Add center annotation
    # -----------------------
    center_text = ax.text(
        0, 0,
        f"{metric_label[0]}",
        fontsize=label_fontsize + 4,
        fontweight='bold',
        ha='center',
        va='center',
        color='darkgray',
        zorder=1
    )
    
    # -----------------------
    # Add title with optional background
    # -----------------------
    title_obj = ax.set_title(
        f"{title}\n{metric_label} Chord Diagram",
        fontsize=title_fontsize,
        fontweight=title_weight,
        color=title_color,
        pad=title_pad
    )
    
    if title_background:
        title_obj.set_bbox(dict(
            facecolor=title_background_color,
            edgecolor='none',
            alpha=title_background_alpha,
            boxstyle="round,pad=0.5"
        ))
    
    if title_halo:
        apply_text_halo(title_obj, halo_width=title_halo_width, halo_color=title_halo_color)
    
    # -----------------------
    # Add colorbar with enhanced styling
    # -----------------------
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=edge_cmap)
    sm.set_array([])
    
    if colorbar_label is None:
        colorbar_label = metric_label
    
    if colorbar_orientation == 'horizontal':
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="horizontal",
            pad=colorbar_pad,
            aspect=colorbar_aspect,
            shrink=colorbar_shrink,
            extend=colorbar_extend
        )
    else:
        cbar = fig.colorbar(
            sm,
            ax=ax,
            orientation="vertical",
            pad=colorbar_pad,
            aspect=colorbar_aspect,
            shrink=colorbar_shrink,
            extend=colorbar_extend
        )
    
    cbar.set_label(colorbar_label, fontsize=colorbar_labelsize)
    cbar.ax.tick_params(labelsize=colorbar_ticksize)
    
    # Set custom ticks if requested
    if colorbar_ticks == 'custom':
        if colorbar_tick_count > 0:
            tick_positions = np.linspace(vmin, vmax, colorbar_tick_count)
            tick_labels = [f"{tick:.2f}" for tick in tick_positions]
            cbar.set_ticks(tick_positions)
            cbar.set_ticklabels(tick_labels)
    
    # -----------------------
    # Add frame if requested
    # -----------------------
    if frame_visible:
        ax.spines['polar'].set_visible(True)
        ax.spines['polar'].set_edgecolor(frame_color)
        ax.spines['polar'].set_linewidth(frame_width)
    
    # -----------------------
    # Apply tight layout if requested
    # -----------------------
    if tight_layout:
        plt.tight_layout(pad=border_padding)
    
    return fig

# ============================================================
# ENHANCED RADAR/SPIDER CHART WITH NUMERICAL LABELS
# ============================================================

def create_enhanced_radar_chart(
    stats,
    metric='correlation',
    colormap_name='coolwarm',
    figsize=(14, 14),
    title="",
    
    # Radar specific parameters
    radar_fill=False,
    radar_fill_alpha=0.2,
    radar_line_width=2.0,
    radar_marker_size=30,
    radar_marker_style='o',
    radar_scale='linear',
    radar_scale_factor=1.0,
    radar_grid_type='polygon',
    radar_grid_levels=3,
    radar_axis_labels=True,
    radar_axis_label_offset=0.1,
    
    # Numerical label controls
    radial_labels=True,
    radial_label_count=5,
    radial_label_fontsize=10,
    radial_label_format='.1f',
    radial_label_color='gray',
    radial_label_weight='normal',
    
    circumferential_labels=True,
    circumferential_label_count=12,
    circumferential_label_fontsize=10,
    circumferential_label_format='.0f°',
    circumferential_label_color='gray',
    circumferential_label_weight='normal',
    
    # Inherited parameters
    label_fontsize=12,
    edge_width_scale=3.0,
    edge_alpha=0.7,
    edge_threshold=0.3,
    edge_curvature=0.35,
    title_fontsize=16,
    background_color='white',
    figure_dpi=100,
    
    # Additional styling
    grid_visible=True,
    grid_alpha=0.2,
    grid_color='gray',
    grid_style='--',
    grid_width=0.5,
    
    axis_visible=True,
    axis_width=1.0,
    axis_color='black',
    axis_alpha=0.5,
    
    value_labels=True,
    value_label_fontsize=8,
    value_label_format='.2f',
    value_label_color='black',
    
    show_legend=False,
    legend_location='upper right',
    legend_fontsize=10,
):
    """
    Create an enhanced radar/spider chart with full numerical label control
    
    Parameters:
    -----------
    All parameters for radar chart visualization
    
    Returns:
    --------
    matplotlib.figure.Figure
        The enhanced radar chart figure
    """
    
    if metric == 'correlation':
        matrix = stats['correlation_matrix']
        vmin, vmax = -1, 1
        metric_label = "Correlation"
    else:
        matrix = stats['covariance_matrix']
        max_abs = np.max(np.abs(matrix))
        vmin, vmax = -max_abs, max_abs
        metric_label = "Covariance"
    
    n = len(FEATURE_COLS)
    means = stats['feature_means']
    
    # Normalize means for radar scale
    if means.max() > means.min():
        norm_means = (means - means.min()) / (means.max() - means.min())
    else:
        norm_means = np.ones(n) * 0.5
    
    # Apply scaling
    if radar_scale == 'log':
        # Avoid log(0)
        norm_means = np.log1p(norm_means * radar_scale_factor)
        norm_means = norm_means / norm_means.max() if norm_means.max() > 0 else norm_means
    elif radar_scale == 'symlog':
        norm_means = np.sign(norm_means) * np.log1p(np.abs(norm_means) * radar_scale_factor)
        norm_means = (norm_means - norm_means.min()) / (norm_means.max() - norm_means.min()) if norm_means.max() > norm_means.min() else norm_means
    else:  # linear
        norm_means = norm_means * radar_scale_factor
    
    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=figure_dpi,
        subplot_kw={'projection': 'polar'}
    )
    
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
    # Setup polar plot
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Get colormap
    cmap = cm.get_cmap(colormap_name)
    
    # ============================================
    # CONFIGURE NUMERICAL LABELS
    # ============================================
    
    # Set radial limits based on data
    max_value = max(1.0, norm_means.max() * 1.1)
    ax.set_ylim(0, max_value)
    
    # Configure radial ticks and labels
    if radial_labels:
        # Calculate radial tick positions
        radial_ticks = np.linspace(0, max_value, radial_label_count + 1)[1:]  # Exclude 0
        
        # Set radial ticks
        ax.set_yticks(radial_ticks)
        
        # Create radial tick labels
        if radar_scale == 'linear':
            # For linear scale, show actual values
            tick_values = radial_ticks / radar_scale_factor * (means.max() - means.min()) + means.min()
            tick_labels = [format(val, radial_label_format) for val in tick_values]
        else:
            # For non-linear scales, show normalized values
            tick_labels = [format(tick, radial_label_format) for tick in radial_ticks]
        
        # Set radial tick labels with styling
        ax.set_yticklabels(
            tick_labels,
            fontsize=radial_label_fontsize,
            color=radial_label_color,
            fontweight=radial_label_weight
        )
        
        # Add radial grid if requested
        if grid_visible:
            for tick in radial_ticks:
                circle = plt.Circle((0, 0), tick, transform=ax.transData._b,
                                  fill=False, alpha=grid_alpha,
                                  edgecolor=grid_color, linestyle=grid_style,
                                  linewidth=grid_width, zorder=1)
                ax.add_patch(circle)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    
    # Configure circumferential ticks and labels
    if circumferential_labels:
        # Set circumferential ticks
        circumferential_angles = np.linspace(0, 2*np.pi, circumferential_label_count, endpoint=False)
        ax.set_xticks(circumferential_angles)
        
        # Create circumferential tick labels (angles in degrees)
        angle_degrees = np.degrees(circumferential_angles)
        tick_labels = [format(angle, circumferential_label_format) for angle in angle_degrees]
        
        # Set circumferential tick labels with styling
        ax.set_xticklabels(
            tick_labels,
            fontsize=circumferential_label_fontsize,
            color=circumferential_label_color,
            fontweight=circumferential_label_weight
        )
        
        # Add circumferential grid if requested
        if grid_visible:
            ax.grid(True, alpha=grid_alpha, color=grid_color,
                   linestyle=grid_style, linewidth=grid_width)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    
    # Add axis lines if requested
    if axis_visible and n > 0:
        for angle in angles:
            ax.plot([angle, angle], [0, max_value],
                   color=axis_color, alpha=axis_alpha,
                   linewidth=axis_width, zorder=2)
    
    # ============================================
    # DRAW CONNECTIONS
    # ============================================
    
    # Draw connections based on correlation/covariance
    connection_colors = []
    connection_widths = []
    connection_alphas = []
    
    for i in range(n):
        for j in range(i + 1, n):
            value = matrix[i, j]
            abs_value = abs(value)
            
            if abs_value > edge_threshold:
                # Calculate positions
                pos_i = (angles[i], norm_means[i])
                pos_j = (angles[j], norm_means[j])
                
                # Calculate color
                color_value = (value - vmin) / (vmax - vmin)
                color = cmap(color_value)
                
                # Calculate width
                width = max(0.5, min(10, abs_value * edge_width_scale))
                
                # Draw connection with curvature
                t = np.linspace(0, 1, 50)
                theta = angles[i] + (angles[j] - angles[i]) * t
                
                # Apply curvature
                if edge_curvature > 0:
                    r = norm_means[i] + (norm_means[j] - norm_means[i]) * t
                    r = r + np.sin(np.pi * t) * edge_curvature * (norm_means[i] + norm_means[j]) / 2
                else:
                    r = norm_means[i] + (norm_means[j] - norm_means[i]) * t
                
                ax.plot(
                    theta, r,
                    color=color,
                    linewidth=width,
                    alpha=edge_alpha,
                    solid_capstyle='round',
                    zorder=5
                )
    
    # ============================================
    # DRAW FEATURE MARKERS AND LABELS
    # ============================================
    
    # Draw feature markers
    for idx, (angle, radius) in enumerate(zip(angles, norm_means)):
        # Main marker
        marker = ax.plot(
            angle,
            radius,
            radar_marker_style,
            markersize=radar_marker_size,
            color=cm.get_cmap('tab20c')(idx),
            markeredgecolor='white',
            markeredgewidth=2,
            zorder=10
        )[0]
        
        # Add feature labels
        if radar_axis_labels:
            label_radius = radius + radar_axis_label_offset
            if label_radius > max_value:
                label_radius = max_value * 0.95
            
            label_angle = angle
            
            # Adjust label orientation
            rotation = np.degrees(label_angle)
            if 90 <= rotation <= 270:
                rotation += 180
                ha = "right"
            else:
                ha = "left"
            
            label_text = NODE_NAMES[idx]
            
            # Add value if requested
            if value_labels:
                value_text = format(means[idx], value_label_format)
                label_text = f"{NODE_NAMES[idx]}\n{value_text}"
            
            txt = ax.text(
                label_angle,
                label_radius,
                label_text,
                fontsize=label_fontsize,
                fontweight='bold',
                rotation=rotation,
                rotation_mode='anchor',
                ha=ha,
                va='center',
                zorder=11
            )
            
            # Add bounding box for better readability
            txt.set_bbox(dict(
                boxstyle="round,pad=0.2",
                facecolor='white',
                edgecolor='lightgray',
                alpha=0.9
            ))
    
    # ============================================
    # DRAW RADAR POLYGON (OPTIONAL)
    # ============================================
    
    if radar_fill and n >= 3:
        # Close the polygon
        poly_angles = np.concatenate([angles, [angles[0]]])
        poly_values = np.concatenate([norm_means, [norm_means[0]]])
        
        # Create filled polygon
        ax.fill(
            poly_angles, poly_values,
            alpha=radar_fill_alpha,
            color='skyblue',
            zorder=3
        )
        
        # Add outline
        ax.plot(
            poly_angles, poly_values,
            color='blue',
            linewidth=radar_line_width,
            alpha=0.7,
            zorder=4
        )
    
    # ============================================
    # ADD GRID BASED ON SELECTED TYPE
    # ============================================
    
    if grid_visible and radar_grid_type == 'polygon' and radar_grid_levels > 0:
        # Create polygon grid
        grid_angles = np.concatenate([angles, [angles[0]]])
        
        for level in range(1, radar_grid_levels + 1):
            grid_value = (max_value / radar_grid_levels) * level
            
            # Create polygon for this level
            poly_values = np.ones(len(grid_angles)) * grid_value
            
            ax.plot(
                grid_angles, poly_values,
                color=grid_color,
                alpha=grid_alpha,
                linestyle=grid_style,
                linewidth=grid_width,
                zorder=1
            )
    
    elif grid_visible and radar_grid_type == 'spiderweb':
        # Create spiderweb grid
        for angle in angles:
            ax.plot([angle, angle], [0, max_value],
                   color=grid_color,
                   alpha=grid_alpha,
                   linestyle=grid_style,
                   linewidth=grid_width,
                   zorder=1)
        
        # Add concentric circles
        for level in range(1, radar_grid_levels + 1):
            grid_value = (max_value / radar_grid_levels) * level
            circle = plt.Circle((0, 0), grid_value, transform=ax.transData._b,
                              fill=False, alpha=grid_alpha,
                              edgecolor=grid_color, linestyle=grid_style,
                              linewidth=grid_width, zorder=1)
            ax.add_patch(circle)
    
    # Set final limits
    ax.set_ylim(0, max_value * 1.1)  # Add 10% padding
    
    # ============================================
    # ADD TITLE AND COLORBAR
    # ============================================
    
    # Add title
    ax.set_title(
        f"{title}\n{metric_label} Radar Chart",
        fontsize=title_fontsize,
        fontweight='bold',
        pad=30
    )
    
    # Add colorbar for connections
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, aspect=50)
    cbar.set_label(f'{metric_label} Value', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=label_fontsize - 2)
    
    # Add legend if requested
    if show_legend:
        # Create custom legend for features
        legend_elements = []
        for idx, feature in enumerate(NODE_NAMES):
            color = cm.get_cmap('tab20c')(idx)
            legend_elements.append(
                plt.Line2D([0], [0], marker='o', color='w',
                          markerfacecolor=color, markersize=10,
                          markeredgecolor='white', markeredgewidth=1,
                          label=feature)
            )
        
        ax.legend(handles=legend_elements, loc=legend_location,
                 fontsize=legend_fontsize, framealpha=0.9)
    
    plt.tight_layout()
    return fig

# ============================================================
# ENHANCED MATRIX HEATMAP
# ============================================================

def create_enhanced_matrix_heatmap(
    stats,
    metric='correlation',
    colormap_name='coolwarm',
    figsize=(12, 10),
    title="",
    
    # Heatmap-specific styling
    cell_padding=0.1,
    cell_rounding=0,
    grid_visible=True,
    grid_color='gray',
    grid_alpha=0.3,
    grid_width=0.5,
    value_threshold=0.3,
    show_values=True,
    value_fontsize=9,
    value_color_threshold=0.7,
    value_format='.2f',
    
    # Numerical axis labels
    xaxis_labels=True,
    xaxis_label_fontsize=10,
    xaxis_label_rotation=45,
    yaxis_labels=True,
    yaxis_label_fontsize=10,
    
    # Inherited common parameters
    label_fontsize=12,
    title_fontsize=16,
    title_weight='bold',
    title_color='black',
    background_color='white',
    figure_dpi=100,
):
    """Create an enhanced matrix heatmap visualization"""
    
    if metric == 'correlation':
        matrix = stats['correlation_matrix']
        vmin, vmax = -1, 1
        metric_label = "Correlation"
    else:
        matrix = stats['covariance_matrix']
        max_abs = np.max(np.abs(matrix))
        vmin, vmax = -max_abs, max_abs
        metric_label = "Covariance"
    
    n = len(FEATURE_COLS)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=figure_dpi)
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
    # Create heatmap with enhanced styling
    im = ax.imshow(
        matrix,
        cmap=colormap_name,
        vmin=vmin,
        vmax=vmax,
        aspect='auto',
        interpolation='nearest'
    )
    
    # Add feature labels
    if xaxis_labels:
        ax.set_xticks(range(n))
        ax.set_xticklabels(FEATURE_COLS, rotation=xaxis_label_rotation,
                          ha='right', fontsize=xaxis_label_fontsize)
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    
    if yaxis_labels:
        ax.set_yticks(range(n))
        ax.set_yticklabels(FEATURE_COLS, fontsize=yaxis_label_fontsize)
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    
    # Add grid if requested
    if grid_visible:
        ax.set_xticks(np.arange(-.5, n, 1), minor=True)
        ax.set_yticks(np.arange(-.5, n, 1), minor=True)
        ax.grid(
            which='minor',
            color=grid_color,
            linestyle='-',
            linewidth=grid_width,
            alpha=grid_alpha
        )
    
    # Add values for significant cells
    if show_values:
        for i in range(n):
            for j in range(n):
                value = matrix[i, j]
                if abs(value) > value_threshold:
                    # Determine text color based on cell darkness
                    cell_color = im.cmap(im.norm(value))
                    luminance = 0.299 * cell_color[0] + 0.587 * cell_color[1] + 0.114 * cell_color[2]
                    text_color = 'white' if luminance < value_color_threshold else 'black'
                    
                    ax.text(
                        j, i,
                        format(value, value_format),
                        ha='center',
                        va='center',
                        color=text_color,
                        fontsize=value_fontsize,
                        fontweight='bold'
                    )
    
    # Add title
    ax.set_title(
        f"{title}\n{metric_label} Matrix",
        fontsize=title_fontsize,
        fontweight=title_weight,
        color=title_color,
        pad=20
    )
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label, fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=label_fontsize - 2)
    
    plt.tight_layout()
    return fig

# ============================================================
# ENHANCED RADIAL PLOT WITH NUMERICAL LABELS
# ============================================================

def create_enhanced_radial_plot(
    stats,
    metric='correlation',
    colormap_name='coolwarm',
    figsize=(14, 14),
    title="",
    
    # Radial plot specific with numerical label controls
    radial_layers=3,
    radial_layer_spacing=0.5,
    radial_labels=True,
    radial_label_count=5,
    radial_label_fontsize=10,
    radial_label_format='.1f',
    radial_label_color='gray',
    radial_label_weight='normal',
    radial_label_offset=0.1,
    
    circumferential_labels=True,
    circumferential_label_count=12,
    circumferential_label_fontsize=10,
    circumferential_label_format='.0f°',
    circumferential_label_color='gray',
    circumferential_label_weight='normal',
    
    radial_lines=True,
    radial_line_alpha=0.2,
    radial_line_color='gray',
    radial_line_style='--',
    radial_line_width=0.5,
    
    # Inherited parameters
    label_fontsize=12,
    edge_width_scale=3.0,
    edge_alpha=0.7,
    edge_threshold=0.3,
    edge_curvature=0.35,
    title_fontsize=16,
    background_color='white',
    figure_dpi=100,
    
    # Additional styling
    show_feature_values=True,
    feature_value_format='.2f',
    feature_value_fontsize=8,
    grid_visible=True,
    grid_alpha=0.2,
    grid_color='gray',
    grid_style='--',
    grid_width=0.5,
):
    """Create an enhanced radial plot visualization with numerical labels"""
    
    if metric == 'correlation':
        matrix = stats['correlation_matrix']
        vmin, vmax = -1, 1
        metric_label = "Correlation"
    else:
        matrix = stats['covariance_matrix']
        max_abs = np.max(np.abs(matrix))
        vmin, vmax = -max_abs, max_abs
        metric_label = "Covariance"
    
    n = len(FEATURE_COLS)
    means = stats['feature_means']
    
    # Create figure
    fig, ax = plt.subplots(
        figsize=figsize,
        dpi=figure_dpi,
        subplot_kw={'projection': 'polar'}
    )
    
    fig.patch.set_facecolor(background_color)
    ax.set_facecolor(background_color)
    
    # Setup
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    
    # Calculate angles
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    
    # Get colormap
    cmap = cm.get_cmap(colormap_name)
    
    # ============================================
    # CONFIGURE NUMERICAL LABELS
    # ============================================
    
    # Set radial limits
    max_radius = radial_layers * radial_layer_spacing
    ax.set_ylim(0, max_radius * 1.1)  # Add 10% padding
    
    # Configure radial ticks and labels
    if radial_labels:
        # Calculate radial tick positions
        radial_ticks = np.linspace(0, max_radius, radial_label_count + 1)[1:]  # Exclude 0
        
        # Set radial ticks
        ax.set_yticks(radial_ticks)
        
        # Create radial tick labels
        tick_labels = [format(tick, radial_label_format) for tick in radial_ticks]
        
        # Set radial tick labels with styling
        ax.set_yticklabels(
            tick_labels,
            fontsize=radial_label_fontsize,
            color=radial_label_color,
            fontweight=radial_label_weight
        )
    else:
        ax.set_yticks([])
        ax.set_yticklabels([])
    
    # Configure circumferential ticks and labels
    if circumferential_labels:
        # Set circumferential ticks
        circumferential_angles = np.linspace(0, 2*np.pi, circumferential_label_count, endpoint=False)
        ax.set_xticks(circumferential_angles)
        
        # Create circumferential tick labels (angles in degrees)
        angle_degrees = np.degrees(circumferential_angles)
        tick_labels = [format(angle, circumferential_label_format) for angle in angle_degrees]
        
        # Set circumferential tick labels with styling
        ax.set_xticklabels(
            tick_labels,
            fontsize=circumferential_label_fontsize,
            color=circumferential_label_color,
            fontweight=circumferential_label_weight
        )
    else:
        ax.set_xticks([])
        ax.set_xticklabels([])
    
    # ============================================
    # CREATE RADIAL GRID
    # ============================================
    
    # Create radial grid
    radii = np.linspace(radial_layer_spacing, radial_layers * radial_layer_spacing, radial_layers)
    
    # Draw radial lines if requested
    if radial_lines:
        for r in radii:
            circle = plt.Circle((0, 0), r, transform=ax.transData._b,
                              fill=False, alpha=radial_line_alpha,
                              edgecolor=radial_line_color,
                              linestyle=radial_line_style,
                              linewidth=radial_line_width,
                              zorder=1)
            ax.add_patch(circle)
    
    # Add grid if requested
    if grid_visible:
        ax.grid(True, alpha=grid_alpha, color=grid_color,
               linestyle=grid_style, linewidth=grid_width)
    
    # ============================================
    # DRAW CONNECTIONS
    # ============================================
    
    # Draw connections
    for i in range(n):
        for j in range(i + 1, n):
            value = matrix[i, j]
            abs_value = abs(value)
            
            if abs_value > edge_threshold:
                # Calculate color
                color = cmap((value - vmin) / (vmax - vmin))
                
                # Calculate position based on value magnitude
                layer_idx = min(int(abs_value * radial_layers), radial_layers - 1)
                r_i = radii[layer_idx]
                r_j = radii[layer_idx]
                
                # Apply curvature if requested
                if edge_curvature > 0:
                    t = np.linspace(0, 1, 30)
                    theta = angles[i] + (angles[j] - angles[i]) * t
                    r = r_i + (r_j - r_i) * t
                    r = r + np.sin(np.pi * t) * edge_curvature * (r_i + r_j) / 2
                    
                    ax.plot(
                        theta, r,
                        color=color,
                        linewidth=abs_value * edge_width_scale,
                        alpha=edge_alpha,
                        solid_capstyle='round',
                        zorder=3
                    )
                else:
                    # Draw straight line
                    ax.plot(
                        [angles[i], angles[j]],
                        [r_i, r_j],
                        color=color,
                        linewidth=abs_value * edge_width_scale,
                        alpha=edge_alpha,
                        solid_capstyle='round',
                        zorder=3
                    )
    
    # ============================================
    # ADD FEATURE MARKERS AND LABELS
    # ============================================
    
    # Add feature markers
    for idx, (angle, radius) in enumerate(zip(angles, [radii[-1]] * n)):
        # Main marker
        ax.plot(
            angle,
            radius,
            'o',
            markersize=20,
            color=cm.get_cmap('tab20c')(idx),
            markeredgecolor='white',
            markeredgewidth=2,
            zorder=10
        )
        
        # Add labels
        label_radius = radius + radial_label_offset
        label_angle = angle
        
        # Adjust label orientation
        rotation = np.degrees(label_angle)
        if 90 <= rotation <= 270:
            rotation += 180
            ha = "right"
        else:
            ha = "left"
        
        label_text = NODE_NAMES[idx]
        
        # Add value if requested
        if show_feature_values:
            value_text = format(means[idx], feature_value_format)
            label_text = f"{NODE_NAMES[idx]}\n{value_text}"
        
        txt = ax.text(
            label_angle,
            label_radius,
            label_text,
            fontsize=label_fontsize,
            fontweight='bold',
            rotation=rotation,
            rotation_mode='anchor',
            ha=ha,
            va='center',
            zorder=11
        )
        
        # Add bounding box
        txt.set_bbox(dict(
            boxstyle="round,pad=0.2",
            facecolor='white',
            edgecolor='lightgray',
            alpha=0.9
        ))
    
    # ============================================
    # ADD TITLE AND COLORBAR
    # ============================================
    
    # Set final limits
    ax.set_ylim(0, radii[-1] + radial_label_offset + 0.5)
    
    # Add title
    ax.set_title(
        f"{title}\n{metric_label} Radial Plot",
        fontsize=title_fontsize,
        fontweight='bold',
        pad=30
    )
    
    # Add colorbar
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    
    cbar = fig.colorbar(sm, ax=ax, orientation='horizontal', pad=0.1, aspect=50)
    cbar.set_label(f'{metric_label} Value', fontsize=label_fontsize)
    cbar.ax.tick_params(labelsize=label_fontsize - 2)
    
    plt.tight_layout()
    return fig

# ============================================================
# STATISTICS PANEL
# ============================================================

def create_statistics_panel(stats, num_rows):
    """Create a statistics summary panel"""
    n_features = len(FEATURE_COLS)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Feature means
    ax1 = axes[0, 0]
    means = stats['feature_means']
    colors = cm.get_cmap('viridis')(np.linspace(0, 1, n_features))
    bars1 = ax1.barh(range(n_features), means, color=colors)
    ax1.set_yticks(range(n_features))
    ax1.set_yticklabels(FEATURE_COLS)
    ax1.set_xlabel('Mean Value')
    ax1.set_title('Feature Means')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars1, means)):
        ax1.text(mean + 0.01, bar.get_y() + bar.get_height()/2,
                f'{mean:.3f}', ha='left', va='center', fontsize=9)
    
    # Plot 2: Feature standard deviations
    ax2 = axes[0, 1]
    stds = stats['feature_stds']
    colors = cm.get_cmap('plasma')(np.linspace(0, 1, n_features))
    bars2 = ax2.barh(range(n_features), stds, color=colors)
    ax2.set_yticks(range(n_features))
    ax2.set_yticklabels(FEATURE_COLS)
    ax2.set_xlabel('Standard Deviation')
    ax2.set_title('Feature Variability')
    ax2.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, std) in enumerate(zip(bars2, stds)):
        ax2.text(std + 0.01, bar.get_y() + bar.get_height()/2,
                f'{std:.3f}', ha='left', va='center', fontsize=9)
    
    # Plot 3: Correlation distribution
    ax3 = axes[1, 0]
    corr_flat = stats['correlation_matrix'][np.triu_indices(n_features, k=1)]
    ax3.hist(corr_flat, bins=20, color='skyblue', edgecolor='black', alpha=0.7)
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel('Correlation Coefficient')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Correlations')
    ax3.grid(True, alpha=0.3)
    
    # Add statistics text
    corr_stats = f"Mean: {corr_flat.mean():.3f}\nStd: {corr_flat.std():.3f}"
    ax3.text(0.05, 0.95, corr_stats, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 4: Top correlations
    ax4 = axes[1, 1]
    # Find top correlations
    corr_pairs = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            value = stats['correlation_matrix'][i, j]
            corr_pairs.append((i, j, abs(value)))
    
    corr_pairs.sort(key=lambda x: x[2], reverse=True)
    top_n = min(10, len(corr_pairs))
    
    top_labels = []
    top_values = []
    for i, j, val in corr_pairs[:top_n]:
        top_labels.append(f"{FEATURE_COLS[i]}-{FEATURE_COLS[j]}")
        top_values.append(val)
    
    colors = cm.get_cmap('coolwarm')(np.linspace(0, 1, top_n))
    bars4 = ax4.barh(range(top_n), top_values, color=colors)
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels(top_labels, fontsize=9)
    ax4.set_xlabel('Absolute Correlation')
    ax4.set_title(f'Top {top_n} Correlations')
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars4, top_values)):
        ax4.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', ha='left', va='center', fontsize=9)
    
    fig.suptitle(f'Statistical Summary ({num_rows} rows)', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    return fig

# ============================================================
# MAIN STREAMLIT APPLICATION
# ============================================================

def main():
    st.set_page_config(
        page_title="Advanced Feature Correlation Explorer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton > button {
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    div[data-baseweb="select"] {
        min-width: 200px;
    }
    .stSlider > div > div > div {
        background: linear-gradient(90deg, #4CAF50, #2196F3);
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .stExpander {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 10px;
        margin-bottom: 10px;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Title and description
    st.title("📊 Advanced Feature Correlation Explorer")
    st.markdown("""
    Visualize statistical relationships between features with **complete visual parameterization**.
    Adjust every aspect of the visualization in real-time for publication-quality results.
    """)
    
    # ============================================================
    # SIDEBAR CONFIGURATION
    # ============================================================
    with st.sidebar:
        st.header("📁 Data Configuration")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Upload CSV File",
            type=["csv"],
            help="Maximum file size: 100MB"
        )
        
        # Data source selection
        if uploaded_file is not None:
            use_example = False
            st.success(f"✅ File uploaded: {uploaded_file.name}")
        else:
            use_example = True
            st.info("📋 Using example dataset")
        
        # Load data
        df = load_data(uploaded_file, use_example)
        
        if df is None:
            st.error("Failed to load data. Please check your file format.")
            st.stop()
        
        # Validate columns
        missing_cols = [col for col in FEATURE_COLS if col not in df.columns]
        if missing_cols:
            st.error(f"Missing required columns: {', '.join(missing_cols)}")
            st.info(f"Available columns: {', '.join(df.columns.tolist())}")
            st.stop()
        
        st.success(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
        
        # Data selection
        st.header("📈 Data Selection")
        
        selection_method = st.radio(
            "Selection Method",
            ["First N rows", "Random sample", "All rows", "Custom range"],
            index=0
        )
        
        if selection_method == "First N rows":
            max_rows = st.slider("Number of rows", 1, min(1000, len(df)), min(100, len(df)))
            selected_indices = list(range(min(max_rows, len(df))))
        elif selection_method == "Random sample":
            sample_size = st.slider("Sample size", 1, min(500, len(df)), min(100, len(df)))
            selected_indices = np.random.choice(len(df), min(sample_size, len(df)), replace=False).tolist()
        elif selection_method == "Custom range":
            start_idx = st.number_input("Start row", 0, len(df)-1, 0)
            end_idx = st.number_input("End row", start_idx, len(df)-1, min(start_idx+100, len(df)-1))
            selected_indices = list(range(start_idx, end_idx + 1))
        else:  # All rows
            selected_indices = list(range(len(df)))
        
        df_selected = df.iloc[selected_indices].copy()
        
        st.info(f"📊 Selected {len(df_selected)} rows for analysis")
        
        # ============================================================
        # VISUALIZATION SELECTION
        # ============================================================
        st.header("🎨 Visualization Type")
        
        viz_type = st.selectbox(
            "Select Visualization",
            ["Enhanced Chord Diagram", "Radar/Spider Chart", "Matrix Heatmap", "Radial Plot", "Statistics Dashboard"],
            index=0
        )
        
        # ============================================================
        # BASIC SETTINGS
        # ============================================================
        st.header("⚙️ Basic Settings")
        
        # Colormap selection with category filtering
        colormap_category = st.selectbox(
            "Colormap Category",
            list(COLORMAP_CATEGORIES.keys()),
            index=0
        )
        
        colormap_options = COLORMAP_CATEGORIES[colormap_category]
        colormap_name = st.selectbox(
            "Colormap",
            colormap_options,
            index=0
        )
        
        # Show colormap preview
        try:
            cmap_preview = plt.get_cmap(colormap_name)
            colors = [cmap_preview(i) for i in np.linspace(0, 1, 20)]
            
            # Create HTML preview
            color_html = '<div style="display: flex; height: 25px; border-radius: 4px; overflow: hidden; margin: 10px 0; border: 1px solid #ddd;">'
            for c in colors:
                r, g, b = int(c[0]*255), int(c[1]*255), int(c[2]*255)
                color_html += f'<div style="flex:1; background-color: rgb({r},{g},{b});"></div>'
            color_html += '</div>'
            
            st.markdown("**Preview:**")
            st.markdown(color_html, unsafe_allow_html=True)
        except:
            pass
        
        # Figure size
        col1, col2 = st.columns(2)
        with col1:
            fig_width = st.slider("Figure Width", 8, 24, 14)
        with col2:
            fig_height = st.slider("Figure Height", 8, 24, 14)
        
        # Metric selection
        metric = st.radio(
            "Connection Metric",
            ["correlation", "covariance"],
            index=0
        )
        
        # ============================================================
        # ADVANCED VISUAL CONTROLS
        # ============================================================
        st.header("🎯 Advanced Styling")
        
        # Initialize default values for all parameters
        # This prevents "not defined" errors
        node_edge_color = "#FFFFFF"
        node_shadow_offset = 0.05
        node_shadow_alpha = 0.3
        node_shadow_color = "#000000"
        node_glow_width = 10
        node_glow_alpha = 0.2
        node_glow_color = "#FFFFFF"
        
        edge_shadow_offset = 0.02
        edge_shadow_alpha = 0.2
        edge_shadow_color = "#000000"
        edge_glow_width = 5
        edge_glow_alpha = 0.15
        edge_glow_color = "#FFFFFF"
        
        radial_grid_color = "#808080"
        radial_grid_style = "--"
        radial_grid_width = 0.5
        
        title_background_color = "#FFFFFF"
        title_halo_width = 2
        title_halo_color = "#FFFFFF"
        
        label_bbox_fc = "#FFFFFF"
        label_bbox_ec = "#D3D3D3"
        label_color = "#000000"
        
        background_color = "#FFFFFF"
        frame_color = "#000000"
        
        # Use expanders for different categories
        with st.expander("📐 Layout & Spacing", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                radial_base = st.slider("Radial Base", 0.5, 3.0, 1.0, 0.1)
                radial_spacing = st.slider("Radial Spacing", 0.1, 2.0, 0.8, 0.1)
                label_offset_radius = st.slider("Label Offset", 0.1, 1.0, 0.35, 0.05)
            with col2:
                subplot_pad = st.slider("Subplot Padding", 0.01, 0.3, 0.12, 0.01)
                label_padding = st.slider("Label Padding", 0.1, 1.0, 0.35, 0.05)
                label_collision_buffer = st.slider("Collision Buffer", 0.05, 0.5, 0.15, 0.05)
        
        with st.expander("🔤 Labels & Text", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                label_fontsize = st.slider("Label Font Size", 8, 24, 12)
                label_bbox_pad = st.slider("Label Box Padding", 0.0, 1.0, 0.25, 0.05)
                label_bbox_rounding = st.slider("Label Box Rounding", 0, 20, 4)
                label_color = st.color_picker("Label Color", label_color)
            with col2:
                title_fontsize = st.slider("Title Font Size", 12, 32, 16)
                colorbar_labelsize = st.slider("Colorbar Label Size", 8, 20, 12)
                colorbar_ticksize = st.slider("Colorbar Tick Size", 6, 18, 10)
                title_color = st.color_picker("Title Color", "#000000")
            
            label_halo = st.checkbox("Label Halo Effect", value=True)
            if label_halo:
                label_halo_width = st.slider("Halo Width", 1, 10, 3)
                label_halo_color = st.color_picker("Halo Color", "#FFFFFF")
            
            label_rotation = st.selectbox(
                "Label Rotation",
                ["auto", "tangential", "horizontal", "radial"],
                index=0
            )
            
            label_show_values = st.checkbox("Show Feature Values", value=True)
            if label_show_values:
                label_value_format = st.selectbox("Value Format", [".0f", ".1f", ".2f", ".3f", ".4f"], index=2)
        
        with st.expander("🔗 Edges & Connections", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                edge_width_scale = st.slider("Edge Width Scale", 0.1, 10.0, 3.0, 0.1)
                edge_min_width = st.slider("Min Edge Width", 0.1, 3.0, 0.5, 0.1)
                edge_max_width = st.slider("Max Edge Width", 1.0, 20.0, 8.0, 0.5)
                edge_threshold = st.slider("Connection Threshold", 0.0, 1.0, 0.3, 0.05)
            with col2:
                edge_curvature = st.slider("Edge Curvature", 0.0, 1.0, 0.35, 0.05)
                edge_alpha = st.slider("Edge Transparency", 0.1, 1.0, 0.7, 0.05)
                edge_smoothness = st.slider("Edge Smoothness", 10, 200, 50)
                max_connections = st.slider("Max Connections", 10, 500, 100)
            
            bezier_control = st.selectbox(
                "Bezier Control",
                ["midpoint", "adaptive", "fixed"],
                index=0
            )
            
            edge_glow = st.checkbox("Edge Glow Effect", value=False)
            if edge_glow:
                edge_glow_width = st.slider("Glow Width", 1, 20, 5)
                edge_glow_alpha = st.slider("Glow Alpha", 0.05, 0.5, 0.15, 0.05)
                edge_glow_color = st.color_picker("Glow Color", "#FFFFFF")
        
        with st.expander("⚫ Nodes & Markers", expanded=False):
            col1, col2 = st.columns(2)
            with col1:
                node_edge_width = st.slider("Node Edge Width", 0.5, 10.0, 2.0, 0.5)
                node_alpha = st.slider("Node Transparency", 0.1, 1.0, 0.9, 0.05)
                node_edge_color = st.color_picker("Node Edge Color", node_edge_color)
                node_shadow = st.checkbox("Node Shadow", value=False)
                if node_shadow:
                    node_shadow_offset = st.slider("Shadow Offset", 0.01, 0.2, node_shadow_offset, 0.01)
                    node_shadow_alpha = st.slider("Shadow Alpha", 0.05, 0.5, node_shadow_alpha, 0.05)
                    node_shadow_color = st.color_picker("Shadow Color", node_shadow_color)
            with col2:
                node_glow = st.checkbox("Node Glow Effect", value=False)
                if node_glow:
                    node_glow_width = st.slider("Glow Width", 1, 20, node_glow_width, 1)
                    node_glow_alpha = st.slider("Glow Alpha", 0.05, 0.5, node_glow_alpha, 0.05)
                    node_glow_color = st.color_picker("Glow Color", node_glow_color)
                
                gradient_nodes = st.checkbox("Gradient Nodes", value=True)
                gradient_edges = st.checkbox("Gradient Edges", value=True)
        
        # Special controls for Radar/Radial plots
        if viz_type in ["Radar/Spider Chart", "Radial Plot"]:
            with st.expander("🎯 Radar/Radial Controls", expanded=True):
                col1, col2 = st.columns(2)
                with col1:
                    radial_labels = st.checkbox("Show Radial Labels", value=True)
                    if radial_labels:
                        radial_label_count = st.slider("Radial Label Count", 2, 10, 5)
                        radial_label_fontsize = st.slider("Radial Label Font Size", 6, 16, 10)
                        radial_label_format = st.selectbox("Radial Label Format", [".0f", ".1f", ".2f", ".3f"], index=1)
                        radial_label_color = st.color_picker("Radial Label Color", "#808080")
                    
                    radial_layers = st.slider("Radial Layers", 1, 10, 3)
                    radial_layer_spacing = st.slider("Layer Spacing", 0.1, 2.0, 0.5, 0.1)
                
                with col2:
                    circumferential_labels = st.checkbox("Show Circumferential Labels", value=True)
                    if circumferential_labels:
                        circumferential_label_count = st.slider("Circumferential Label Count", 4, 36, 12)
                        circumferential_label_fontsize = st.slider("Circumferential Label Font Size", 6, 16, 10)
                        circumferential_label_format = st.selectbox("Circumferential Label Format", [".0f°", ".1f°", ".2f°"], index=0)
                        circumferential_label_color = st.color_picker("Circumferential Label Color", "#808080")
                    
                    radar_fill = st.checkbox("Fill Radar Polygon", value=False)
                    if radar_fill:
                        radar_fill_alpha = st.slider("Fill Alpha", 0.1, 1.0, 0.2, 0.05)
        
        with st.expander("🎨 Colors & Effects", expanded=False):
            background_color = st.color_picker("Background Color", background_color)
            frame_color = st.color_picker("Frame Color", frame_color)
            
            label_bbox_fc = st.color_picker("Label Box Color", label_bbox_fc)
            label_bbox_ec = st.color_picker("Label Box Edge", label_bbox_ec)
            
            label_bbox_alpha = st.slider("Label Box Alpha", 0.0, 1.0, 0.85, 0.05)
            label_bbox_lw = st.slider("Label Box Line Width", 0.0, 3.0, 1.0, 0.1)
            
            frame_visible = st.checkbox("Show Frame", value=False)
            if frame_visible:
                frame_width = st.slider("Frame Width", 0.5, 5.0, 1.0, 0.1)
        
        with st.expander("⚡ Performance", expanded=False):
            use_caching = st.checkbox("Use Caching", value=True)
            figure_dpi = st.slider("Figure DPI", 72, 300, 100)
            anti_aliasing = st.checkbox("Anti-Aliasing", value=True)
            label_collision_avoidance = st.checkbox("Label Collision Avoidance", value=True)
            
            clear_cache = st.button("Clear Cache")
            if clear_cache:
                st.cache_data.clear()
                st.success("Cache cleared!")
        
        # ============================================================
        # EXPORT SETTINGS
        # ============================================================
        st.header("💾 Export")
        export_dpi = st.slider("Export DPI", 72, 600, 300)
        show_download = st.checkbox("Show Download Options", value=True)
    
    # ============================================================
    # MAIN CONTENT AREA
    # ============================================================
    tab1, tab2, tab3 = st.tabs(["📊 Visualization", "📈 Statistics", "📋 Data"])
    
    with tab1:
        st.header(f"{viz_type}")
        
        # Compute statistics
        with st.spinner("Computing statistics..."):
            stats = compute_statistics(df_selected)
        
        # Create visualization based on selection
        with st.spinner("Generating visualization..."):
            try:
                if viz_type == "Enhanced Chord Diagram":
                    fig = create_enhanced_chord_diagram(
                        stats=stats,
                        metric=metric,
                        colormap_name=colormap_name,
                        edge_threshold=edge_threshold,
                        max_connections=max_connections,
                        figsize=(fig_width, fig_height),
                        title=f"Rows: {len(df_selected)}",
                        
                        # Node styling
                        node_edge_width=node_edge_width,
                        node_alpha=node_alpha,
                        node_edge_color=node_edge_color,
                        node_shadow=node_shadow,
                        node_shadow_offset=node_shadow_offset,
                        node_shadow_alpha=node_shadow_alpha,
                        node_shadow_color=node_shadow_color,
                        node_glow=node_glow,
                        node_glow_width=node_glow_width,
                        node_glow_alpha=node_glow_alpha,
                        node_glow_color=node_glow_color,
                        gradient_nodes=gradient_nodes,
                        
                        # Label styling
                        label_fontsize=label_fontsize,
                        label_padding=label_padding,
                        label_offset_radius=label_offset_radius,
                        label_bbox_pad=label_bbox_pad,
                        label_bbox_alpha=label_bbox_alpha,
                        label_bbox_fc=label_bbox_fc,
                        label_bbox_ec=label_bbox_ec,
                        label_bbox_lw=label_bbox_lw,
                        label_bbox_rounding=label_bbox_rounding,
                        label_halo=label_halo,
                        label_halo_width=label_halo_width,
                        label_halo_color=label_halo_color,
                        label_rotation=label_rotation,
                        label_color=label_color,
                        label_show_values=label_show_values,
                        label_value_format=label_value_format,
                        
                        # Edge styling
                        edge_alpha=edge_alpha,
                        edge_width_scale=edge_width_scale,
                        edge_min_width=edge_min_width,
                        edge_max_width=edge_max_width,
                        edge_curvature=edge_curvature,
                        edge_smoothness=edge_smoothness,
                        edge_glow=edge_glow,
                        edge_glow_width=edge_glow_width,
                        edge_glow_alpha=edge_glow_alpha,
                        edge_glow_color=edge_glow_color,
                        gradient_edges=gradient_edges,
                        bezier_control=bezier_control,
                        
                        # Radial layout
                        radial_base=radial_base,
                        radial_spacing=radial_spacing,
                        radial_label_offset=label_offset_radius,
                        
                        # Title and text
                        title_fontsize=title_fontsize,
                        title_color=title_color,
                        
                        # Colorbar
                        colorbar_labelsize=colorbar_labelsize,
                        colorbar_ticksize=colorbar_ticksize,
                        colorbar_pad=subplot_pad,
                        
                        # Layout
                        subplot_pad=subplot_pad,
                        background_color=background_color,
                        figure_dpi=figure_dpi,
                        frame_visible=frame_visible,
                        frame_color=frame_color,
                        frame_width=frame_width,
                        
                        # Advanced
                        anti_aliasing=anti_aliasing,
                        label_collision_avoidance=label_collision_avoidance,
                        label_collision_buffer=label_collision_buffer,
                    )
                
                elif viz_type == "Radar/Spider Chart":
                    fig = create_enhanced_radar_chart(
                        stats=stats,
                        metric=metric,
                        colormap_name=colormap_name,
                        figsize=(fig_width, fig_height),
                        title=f"Rows: {len(df_selected)}",
                        
                        # Radar specific
                        radar_fill=radar_fill,
                        radar_fill_alpha=radar_fill_alpha if radar_fill else 0.2,
                        radar_line_width=2.0,
                        radar_marker_size=30,
                        radar_scale='linear',
                        radar_scale_factor=1.0,
                        radar_grid_type='polygon',
                        radar_grid_levels=3,
                        radar_axis_labels=True,
                        radar_axis_label_offset=0.1,
                        
                        # Numerical label controls
                        radial_labels=radial_labels,
                        radial_label_count=radial_label_count,
                        radial_label_fontsize=radial_label_fontsize,
                        radial_label_format=radial_label_format,
                        radial_label_color=radial_label_color,
                        
                        circumferential_labels=circumferential_labels,
                        circumferential_label_count=circumferential_label_count,
                        circumferential_label_fontsize=circumferential_label_fontsize,
                        circumferential_label_format=circumferential_label_format,
                        circumferential_label_color=circumferential_label_color,
                        
                        # Other parameters
                        label_fontsize=label_fontsize,
                        edge_width_scale=edge_width_scale,
                        edge_alpha=edge_alpha,
                        edge_threshold=edge_threshold,
                        edge_curvature=edge_curvature,
                        title_fontsize=title_fontsize,
                        background_color=background_color,
                        figure_dpi=figure_dpi,
                    )
                
                elif viz_type == "Matrix Heatmap":
                    fig = create_enhanced_matrix_heatmap(
                        stats=stats,
                        metric=metric,
                        colormap_name=colormap_name,
                        figsize=(fig_width, fig_height),
                        title=f"Rows: {len(df_selected)}",
                        label_fontsize=label_fontsize,
                        title_fontsize=title_fontsize,
                        title_color=title_color,
                        background_color=background_color,
                        figure_dpi=figure_dpi,
                    )
                
                elif viz_type == "Radial Plot":
                    fig = create_enhanced_radial_plot(
                        stats=stats,
                        metric=metric,
                        colormap_name=colormap_name,
                        figsize=(fig_width, fig_height),
                        title=f"Rows: {len(df_selected)}",
                        
                        # Radial plot specific
                        radial_layers=radial_layers,
                        radial_layer_spacing=radial_layer_spacing,
                        radial_labels=radial_labels,
                        radial_label_count=radial_label_count,
                        radial_label_fontsize=radial_label_fontsize,
                        radial_label_format=radial_label_format,
                        radial_label_color=radial_label_color,
                        radial_label_offset=0.1,
                        
                        circumferential_labels=circumferential_labels,
                        circumferential_label_count=circumferential_label_count,
                        circumferential_label_fontsize=circumferential_label_fontsize,
                        circumferential_label_format=circumferential_label_format,
                        circumferential_label_color=circumferential_label_color,
                        
                        # Other parameters
                        label_fontsize=label_fontsize,
                        edge_width_scale=edge_width_scale,
                        edge_alpha=edge_alpha,
                        edge_threshold=edge_threshold,
                        edge_curvature=edge_curvature,
                        title_fontsize=title_fontsize,
                        background_color=background_color,
                        figure_dpi=figure_dpi,
                    )
                
                else:  # Statistics Dashboard
                    # Create multiple visualizations in a dashboard
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig1 = create_enhanced_chord_diagram(
                            stats=stats,
                            metric=metric,
                            colormap_name=colormap_name,
                            figsize=(fig_width/2, fig_height/2),
                            title=f"Chord Diagram",
                            label_fontsize=max(8, label_fontsize-2),
                        )
                        st.pyplot(fig1, use_container_width=True)
                        plt.close(fig1)
                    
                    with col2:
                        fig2 = create_enhanced_matrix_heatmap(
                            stats=stats,
                            metric=metric,
                            colormap_name=colormap_name,
                            figsize=(fig_width/2, fig_height/2),
                            title=f"Matrix Heatmap",
                            label_fontsize=max(8, label_fontsize-2),
                        )
                        st.pyplot(fig2, use_container_width=True)
                        plt.close(fig2)
                    
                    # Don't create a single figure for dashboard
                    fig = None
                
                # Display the figure (if not dashboard)
                if fig is not None:
                    st.pyplot(fig, use_container_width=True)
                    
                    # Download options
                    if show_download:
                        st.subheader("📥 Export Options")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            # PNG download
                            buf_png = io.BytesIO()
                            fig.savefig(buf_png, format='png', dpi=export_dpi,
                                       bbox_inches='tight', facecolor=background_color)
                            buf_png.seek(0)
                            st.download_button(
                                label="📥 PNG",
                                data=buf_png,
                                file_name=f"correlation_{viz_type.replace('/', '_').lower()}_{metric}.png",
                                mime="image/png",
                                help="High-quality PNG image"
                            )
                        
                        with col2:
                            # PDF download
                            buf_pdf = io.BytesIO()
                            fig.savefig(buf_pdf, format='pdf',
                                       bbox_inches='tight', facecolor=background_color)
                            buf_pdf.seek(0)
                            st.download_button(
                                label="📥 PDF",
                                data=buf_pdf,
                                file_name=f"correlation_{viz_type.replace('/', '_').lower()}_{metric}.pdf",
                                mime="application/pdf",
                                help="Vector PDF for publications"
                            )
                        
                        with col3:
                            # SVG download
                            buf_svg = io.BytesIO()
                            fig.savefig(buf_svg, format='svg',
                                       bbox_inches='tight', facecolor=background_color)
                            buf_svg.seek(0)
                            st.download_button(
                                label="📥 SVG",
                                data=buf_svg,
                                file_name=f"correlation_{viz_type.replace('/', '_').lower()}_{metric}.svg",
                                mime="image/svg+xml",
                                help="Scalable Vector Graphics"
                            )
                        
                        with col4:
                            # TIFF download
                            buf_tiff = io.BytesIO()
                            fig.savefig(buf_tiff, format='tiff', dpi=export_dpi,
                                       bbox_inches='tight', facecolor=background_color)
                            buf_tiff.seek(0)
                            st.download_button(
                                label="📥 TIFF",
                                data=buf_tiff,
                                file_name=f"correlation_{viz_type.replace('/', '_').lower()}_{metric}.tiff",
                                mime="image/tiff",
                                help="High-resolution TIFF for printing"
                            )
                    
                    # Clean up
                    plt.close(fig)
                    gc.collect()
                
            except Exception as e:
                st.error(f"Error creating visualization: {str(e)}")
                import traceback
                with st.expander("Error Details"):
                    st.code(traceback.format_exc())
    
    with tab2:
        st.header("Statistical Summary")
        
        # Compute correlation and covariance matrices
        corr_matrix = df_selected[FEATURE_COLS].corr().round(3)
        cov_matrix = df_selected[FEATURE_COLS].cov().round(3)
        
        # Display in columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Correlation Matrix")
            
            # Style correlation matrix
            def color_corr(val):
                if val < -0.7:
                    return 'background-color: #dc2626; color: white;'  # Red
                elif val < -0.3:
                    return 'background-color: #fb923c; color: white;'  # Orange
                elif val < 0.3:
                    return 'background-color: #4ade80; color: black;'  # Light green
                elif val < 0.7:
                    return 'background-color: #16a34a; color: white;'  # Green
                else:
                    return 'background-color: #14532d; color: white;'  # Dark green
            
            st.dataframe(corr_matrix.style.applymap(color_corr), 
                        use_container_width=True,
                        height=400)
        
        with col2:
            st.subheader("📈 Covariance Matrix")
            st.dataframe(cov_matrix, use_container_width=True, height=400)
        
        # Feature statistics
        st.subheader("📋 Feature Statistics")
        stats_df = pd.DataFrame({
            'Mean': df_selected[FEATURE_COLS].mean(),
            'Std': df_selected[FEATURE_COLS].std(),
            'Min': df_selected[FEATURE_COLS].min(),
            'Max': df_selected[FEATURE_COLS].max(),
            'Median': df_selected[FEATURE_COLS].median(),
            'CV': df_selected[FEATURE_COLS].std() / df_selected[FEATURE_COLS].mean()
        }).round(3)
        
        st.dataframe(stats_df, use_container_width=True)
        
        # Strongest correlations
        st.subheader("🔗 Strongest Relationships")
        
        # Find top correlations
        corr_pairs = []
        for i in range(len(FEATURE_COLS)):
            for j in range(i + 1, len(FEATURE_COLS)):
                corr = corr_matrix.iloc[i, j]
                corr_pairs.append((FEATURE_COLS[i], FEATURE_COLS[j], corr))
        
        # Sort by absolute value
        corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)
        
        # Display in columns
        cols = st.columns(3)
        for idx, (feature1, feature2, value) in enumerate(corr_pairs[:9]):
            with cols[idx % 3]:
                # Create a metric card
                st.metric(
                    label=f"{feature1} ↔ {feature2}",
                    value=f"{value:.3f}",
                    delta="Strong" if abs(value) > 0.7 else 
                          "Moderate" if abs(value) > 0.3 else "Weak",
                    delta_color="normal" if value > 0 else "inverse"
                )
    
    with tab3:
        st.header("Data Preview")
        
        # Show selected data
        st.write(f"Showing {len(df_selected)} selected rows:")
        
        # Interactive data exploration
        show_cols = st.multiselect(
            "Select columns to display",
            df_selected.columns.tolist(),
            default=FEATURE_COLS
        )
        
        # Pagination
        page_size = st.slider("Rows per page", 10, 100, 20)
        total_pages = max(1, len(df_selected) // page_size)
        
        page_num = st.number_input("Page", 1, total_pages, 1)
        start_idx = (page_num - 1) * page_size
        end_idx = min(start_idx + page_size, len(df_selected))
        
        st.dataframe(df_selected[show_cols].iloc[start_idx:end_idx], 
                    use_container_width=True)
        
        # Data summary
        st.subheader("📊 Data Summary")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", len(df_selected))
            st.metric("Features", len(FEATURE_COLS))
        
        with col2:
            st.metric("Memory Usage", 
                     f"{df_selected.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
            st.metric("Missing Values", df_selected[FEATURE_COLS].isnull().sum().sum())
        
        with col3:
            total_mean = df_selected[FEATURE_COLS].mean().mean()
            total_std = df_selected[FEATURE_COLS].std().mean()
            st.metric("Average Feature Mean", f"{total_mean:.3f}")
            st.metric("Average Feature Std", f"{total_std:.3f}")
        
        with col4:
            min_val = df_selected[FEATURE_COLS].min().min()
            max_val = df_selected[FEATURE_COLS].max().max()
            st.metric("Overall Min", f"{min_val:.3f}")
            st.metric("Overall Max", f"{max_val:.3f}")
    
    # ============================================================
    # FOOTER AND DOCUMENTATION
    # ============================================================
    st.markdown("---")
    
    with st.expander("📖 Visualization Guide", expanded=False):
        st.markdown("""
        ## 🎨 Complete Visual Parameterization Guide
        
        ### **Radar/Radial Chart Numerical Labels**
        
        **Radial Labels (Distance from center):**
        - **Show Radial Labels**: Toggle radial numerical labels
        - **Radial Label Count**: Number of concentric circles with labels
        - **Radial Label Font Size**: Font size for radial labels
        - **Radial Label Format**: Number format (e.g., .1f for 1 decimal)
        - **Radial Label Color**: Color of radial labels
        
        **Circumferential Labels (Angles):**
        - **Show Circumferential Labels**: Toggle angular labels
        - **Circumferential Label Count**: Number of angular divisions
        - **Circumferential Label Font Size**: Font size for angular labels
        - **Circumferential Label Format**: Format for angles (e.g., .0f° for degrees)
        - **Circumferential Label Color**: Color of angular labels
        
        ### **Layout & Spacing**
        - **Radial Base**: Distance from center to inner edge of nodes
        - **Radial Spacing**: Overall scale of the radial layout
        - **Label Offset**: Distance from nodes to labels
        - **Label Padding**: Additional space around text in label boxes
        - **Collision Buffer**: Minimum angular distance between labels
        
        ### **Labels & Text**
        - **Label Rotation**: How labels are oriented around the circle
          - *Auto*: Automatically flips labels on bottom half
          - *Tangential*: Labels follow circle tangent
          - *Horizontal*: All labels horizontal
          - *Radial*: Labels point outward from center
        - **Halo Effect**: White outline around text for better readability
        - **Bounding Box**: Background box behind labels
        
        ### **Edges & Connections**
        - **Edge Curvature**: How much edges curve outward (0=straight, 1=maximum)
        - **Bezier Control**: Method for calculating curve control points
        - **Glow Effect**: Adds a soft glow around edges for emphasis
        - **Width Scaling**: Multiplier for edge thickness based on correlation strength
        
        ### **Nodes & Markers**
        - **Gradient Nodes**: Creates color gradient within nodes
        - **Shadow/Glow Effects**: Adds depth and emphasis
        - **Edge Width**: Thickness of node borders
        
        ### **Performance Tips**
        1. Reduce **Max Connections** for faster rendering
        2. Lower **Edge Smoothness** for complex diagrams
        3. Use **Label Collision Avoidance** for crowded diagrams
        4. Export at high DPI for publications
        5. Clear cache if experiencing slowdowns
        
        ### **Export Recommendations**
        - **PNG**: For web use and presentations
        - **PDF**: For publications and vector editing
        - **SVG**: For web graphics and further editing
        - **TIFF**: For high-quality print production
        """)

if __name__ == "__main__":
    # Set environment variables for performance
    import os
    os.environ["PYTHONUNBUFFERED"] = "1"
    os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "100"
    
    # Run the application
    main()
