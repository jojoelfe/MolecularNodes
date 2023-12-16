import bpy
import numpy as np

from ..blender import (
    nodes, coll, obj
)

__all__ = ['load']

bpy.types.Scene.MN_import_star_file_path = bpy.props.StringProperty(
    name = 'File', 
    description = 'File path for the `.star` file to import.', 
    subtype = 'FILE_PATH', 
    maxlen = 0
    )
bpy.types.Scene.MN_import_star_file_name = bpy.props.StringProperty(
    name = 'Name', 
    description = 'Name of the created object.', 
    default = 'NewStarInstances', 
    maxlen = 0
    )
bpy.types.Scene.MN_import_micrograph = bpy.props.BoolProperty(
    name = 'Import Micrograph', 
    description = 'Load micrographs as images and add them to the scene.',
    default = False
    )

def _update_micrograph_texture(obj, mat, star_type):
    import mrcfile
    from pathlib import Path
    if star_type == 'relion':
        micrograph_path = obj['rlnMicrographName_categories'][obj.modifiers['MolecularNodes']["Input_3"] - 1]
    elif star_type == 'cistem':
        micrograph_path = obj['cisTEMOriginalImageFilename_categories'][obj.modifiers['MolecularNodes']["Input_3"] - 1].strip("'")
    else:
        return

    tiff_path = micrograph_path + ".tiff"
    if not Path(tiff_path).exists():
        print("Converting micrograph: ", micrograph_path)
        with mrcfile.open(micrograph_path) as mrc:
            micrograph_data = mrc.data.copy()

        # For 3D data sum over the z axis. Probalby would be nicer to load the data as a volume
        if micrograph_data.ndim == 3:
            micrograph_data = np.sum(micrograph_data, axis=0)

        from PIL import Image
        im = Image.fromarray(micrograph_data[::-1,:])
        im.save(tiff_path)
    im_name = tiff_path.split("/")[-1]
    if im_name not in bpy.data.images:
        image_obj = bpy.data.images.load(tiff_path)
    else:
        image_obj = bpy.data.images[im_name]
    mat.node_tree.nodes['Image Texture'].image = image_obj
    obj.modifiers['MolecularNodes'].node_group.nodes['MOL_micrograph_plane'].inputs['Image'].default_value = image_obj
    bpy.context.view_layer.objects.active = obj
    #bpy.context.space_data.context = 'MODIFIER'



def load(
    file_path, 
    name = 'NewStarInstances', 
    node_tree = True,
    world_scale =  0.01,
    import_micrograph = False,
    ):
    import starfile
    
    star = starfile.read(file_path)
    star_type = None
    
    
    # only RELION 3.1 and cisTEM STAR files are currently supported, fail gracefully
    if isinstance(star, dict) and 'particles' in star and 'optics' in star:
        star_type = 'relion'
    elif "cisTEMAnglePsi" in star:
        star_type = 'cistem'
    else:
        raise ValueError(
        'File is not a valid RELION>=3.1 or cisTEM STAR file, other formats are not currently supported.'
        )
    
    # Get absolute position and orientations    
    if star_type == 'relion':
        df = star['particles'].merge(star['optics'], on='rlnOpticsGroup')

        # get necessary info from dataframes
        # Standard cryoEM starfile don't have rlnCoordinateZ. If this column is not present 
        # Set it to "0"
        if "rlnCoordinateZ" not in df:
            df['rlnCoordinateZ'] = 0
            
        xyz = df[['rlnCoordinateX', 'rlnCoordinateY', 'rlnCoordinateZ']].to_numpy()
        pixel_size = df['rlnImagePixelSize'].to_numpy().reshape((-1, 1))
        xyz = xyz * pixel_size
        shift_column_names = ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']
        if all([col in df.columns for col in shift_column_names]):
            shifts_ang = df[shift_column_names].to_numpy()
            xyz = xyz - shifts_ang 
        df['MNAnglePhi'] = df['rlnAngleRot']
        df['MNAngleTheta'] = df['rlnAngleTilt']
        df['MNAnglePsi'] = df['rlnAnglePsi']
        image_id = df['rlnMicrographName'].astype('category').cat.codes.to_numpy()
        
    elif star_type == 'cistem':
        df = star
        df['cisTEMZFromDefocus'] = (df['cisTEMDefocus1'] + df['cisTEMDefocus2']) / 2
        df['cisTEMZFromDefocus'] = df['cisTEMZFromDefocus'] - df['cisTEMZFromDefocus'].median()
        xyz = df[['cisTEMOriginalXPosition', 'cisTEMOriginalYPosition', 'cisTEMZFromDefocus']].to_numpy()
        df['MNAnglePhi'] = df['cisTEMAnglePhi']
        df['MNAngleTheta'] = df['cisTEMAngleTheta']
        df['MNAnglePsi'] = df['cisTEMAnglePsi']
        image_id = df['cisTEMOriginalImageFilename'].astype('category').cat.codes.to_numpy()

    ensemble = obj.create_object(xyz * world_scale, collection=coll.mn(), name=name)

    ensemble.mn['molecule_type'] = 'star'
    ensemble.mn['star_type'] = star_type


    # create the attribute and add the data for the image id
    obj.add_attribute(ensemble, 'MNImageId', image_id, 'INT', 'POINT')
    
    # create attribute for every column in the STAR file
    for col in df.columns:
        col_type = df[col].dtype    
        # If col_type is numeric directly add
        if np.issubdtype(col_type, np.number):
            obj.add_attribute(ensemble, col, df[col].to_numpy().reshape(-1), 'FLOAT', 'POINT')
        
        # If col_type is object, convert to category and add integer values
        elif col_type == object:
            codes = df[col].astype('category').cat.codes.to_numpy().reshape(-1)
            obj.add_attribute(ensemble, col, codes, 'INT', 'POINT')
            # Add the category names as a property to the blender object
            ensemble[col + '_categories'] = list(df[col].astype('category').cat.categories)
    
    if node_tree:
        nodes.create_starting_nodes_starfile(ensemble)
        
        if import_micrograph:
            mat = nodes.MN_micrograph_material()
            mat.name = name + "_micrograph_material"

            # Setup the size of the micrograph plane
            nodes.add_micrograph_to_starfile_nodes(ensemble, mat)
            _update_micrograph_texture(obj, mat, star_type)
            bpy.app.handlers.depsgraph_update_post.append(lambda x,y: _update_micrograph_texture(obj, mat, star_type))
    
    return ensemble


class MN_OT_Import_Star_File(bpy.types.Operator):
    bl_idname = "mn.import_star_file"
    bl_label = "Load"
    bl_description = "Will import the given file, setting up the points to instance an object."
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        scene = context.scene
        load(
            file_path = scene.MN_import_star_file_path, 
            name = scene.MN_import_star_file_name, 
            node_tree = True,
            import_micrograph = scene.MN_import_micrograph
        )
        return {"FINISHED"}


def panel(layout, scene):
    layout.label(text = "Load Star File", icon='FILE_TICK')
    layout.separator()
    row_import = layout.row()
    row_import.prop(scene, 'MN_import_star_file_name')
    layout.prop(scene, 'MN_import_star_file_path')
    row_import.operator('mn.import_star_file')
    layout.separator()
    layout.label(text = "Options", icon = "MODIFIER")
    options = layout.column(align = True)
    grid = options.grid_flow()  
    grid.prop(scene, 'MN_import_micrograph')
