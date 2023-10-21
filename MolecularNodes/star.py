import bpy
import numpy as np
from . import coll
from . import nodes
from .obj import create_object


bpy.types.Scene.mol_import_star_file_path = bpy.props.StringProperty(
    name = 'star_file_path', 
    description = 'File path for the star file to import.', 
    options = {'TEXTEDIT_UPDATE'}, 
    default = '', 
    subtype = 'FILE_PATH', 
    maxlen = 0
    )
bpy.types.Scene.mol_import_star_file_name = bpy.props.StringProperty(
    name = 'star_file_name', 
    description = 'Name of the created object.', 
    options = {'TEXTEDIT_UPDATE'}, 
    default = 'NewStarInstances', 
    subtype = 'NONE', 
    maxlen = 0
    )


def _update_micrograph_texture(obj, mat, star_type):
    import mrcfile
    from pathlib import Path
    micrograph_path = obj['cisTEMOriginalImageFilename_categories'][obj.modifiers['MolecularNodes']["Input_3"] - 1]
    if star_type == 'relion':
        micrograph_path = obj['rlnMicrographName_categories'][obj.modifiers['MolecularNodes']["Input_3"] - 1]
    elif star_type == 'cistem':
        micrograph_path = obj['cisTEMOriginalImageFilename_categories'][obj.modifiers['MolecularNodes']["Input_3"] - 1]
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
    obj.modifiers['MolecularNodes'].node_group.nodes['Group.001'].inputs['Image'].default_value = image_obj



def load_star_file(
    file_path, 
    obj_name = 'NewStarInstances', 
    node_tree = True,
    world_scale =  0.01,
    load_micrograph = True 
    ):
    import starfile
    from eulerangles import ConversionMeta, convert_eulers
    star = starfile.read(bpy.path.abspath(file_path), always_dict=True)
    
    star_type = None
    # only RELION 3.1 and cisTEM STAR files are currently supported, fail gracefully
    if 'particles' in star and 'optics' in star:
        star_type = 'relion'
    elif "cisTEMAnglePsi" in star[0]:
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
        xyz *= pixel_size
        shift_column_names = ['rlnOriginXAngst', 'rlnOriginYAngst', 'rlnOriginZAngst']
        if all([col in df.columns for col in shift_column_names]):
            shifts_ang = df[shift_column_names].to_numpy()
            xyz -= shifts_ang 
        euler_angles = df[['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']].to_numpy()
        image_id = df['rlnMicrographName'].astype('category').cat.codes.to_numpy()
        
    elif star_type == 'cistem':
        df = star[0]
        df['cisTEMZFromDefocus'] = (df['cisTEMDefocus1'] + df['cisTEMDefocus2']) / 2
        df['cisTEMZFromDefocus'] = df['cisTEMZFromDefocus'] - df['cisTEMZFromDefocus'].median()
        pixel_size = df['cisTEMPixelSize'].to_numpy().reshape((-1, 1))
        xyz = df[['cisTEMOriginalXPosition', 'cisTEMOriginalYPosition', 'cisTEMZFromDefocus']].to_numpy()
        euler_angles = df[['cisTEMAnglePhi', 'cisTEMAngleTheta', 'cisTEMAnglePsi']].to_numpy()
        image_id = df['cisTEMOriginalImageFilename'].astype('category').cat.codes.to_numpy()
    else: 
        return

    # coerce starfile Euler angles to Blender convention
    
    target_metadata = ConversionMeta(name='output', 
                                    axes='xyz', 
                                    intrinsic=False,
                                    right_handed_rotation=True,
                                    active=True)
    eulers = np.deg2rad(convert_eulers(euler_angles, 
                               source_meta='relion', 
                               target_meta=target_metadata))

    obj_name = bpy.path.display_name(file_path)
    obj = create_object(obj_name, coll.mn(), xyz * world_scale)
    
    # vectors have to be added as a 1D array currently
    rotations = eulers.reshape(len(eulers) * 3)
    # create the attribute and add the data for the rotations
    attribute = obj.data.attributes.new('MOLRotation', 'FLOAT_VECTOR', 'POINT')
    attribute.data.foreach_set('vector', rotations)

    # create the attribute and add the data for the image id
    attribute_imgid = obj.data.attributes.new('MOLImageId', 'INT', 'POINT')
    attribute_imgid.data.foreach_set('value', image_id)
    # create attribute for every column in the STAR file
    for col in df.columns:
        col_type = df[col].dtype
        # If col_type is numeric directly add
        if np.issubdtype(col_type, np.number):
            attribute = obj.data.attributes.new(col, 'FLOAT', 'POINT')
            attribute.data.foreach_set('value', df[col].to_numpy().reshape(-1))
        # If col_type is object, convert to category and add integer values
        elif col_type == object:
            attribute = obj.data.attributes.new(col, 'INT', 'POINT')
            codes = df[col].astype('category').cat.codes
            attribute.data.foreach_set('value', codes.to_numpy().reshape(-1))
            # Add the category names as a property to the blender object
            obj[col + '_categories'] = list(df[col].astype('category').cat.categories)
    
    if node_tree:
        node_mod, node_group = nodes.create_starting_nodes_starfile(obj)

        if load_micrograph:
            

            mat = nodes.mol_micrograph_material()
            mat.name = obj_name + "_micrograph_material"
            
            # Setup the size of the micrograph plane
            nodes.add_micrograph_to_starfile_nodes(node_mod, node_group, mat, pixel_size[0], world_scale)
            bpy.app.handlers.depsgraph_update_post.append(lambda x,y: _update_micrograph_texture(obj, mat, star_type))
    return obj


def panel(layout_function, scene):
    col_main = layout_function.column(heading = "", align = False)
    col_main.label(text = "Import Star File")
    row_import = col_main.row()
    row_import.prop(
        bpy.context.scene, 'mol_import_star_file_name', 
        text = 'Name', 
        emboss = True
    )
    col_main.prop(
        bpy.context.scene, 'mol_import_star_file_path', 
        text = '.star File Path', 
        emboss = True
    )
    row_import.operator('mol.import_star_file', text = 'Load', icon = 'FILE_TICK')



class MOL_OT_Import_Star_File(bpy.types.Operator):
    bl_idname = "mol.import_star_file"
    bl_label = "Import Star File"
    bl_description = "Will import the given file, setting up the points to instance an object."
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return True

    def execute(self, context):
        load_star_file(
            file_path = bpy.context.scene.mol_import_star_file_path, 
            obj_name = bpy.context.scene.mol_import_star_file_name, 
            node_tree = True,
            
        )
        return {"FINISHED"}