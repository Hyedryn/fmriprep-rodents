import numpy as np
# Import Node and Workflow object and FSL interface
from nipype import Node, Workflow
from nipype.interfaces import fsl, freesurfer, utility as niu
from niworkflows.interfaces.fixes import FixHeaderApplyTransforms as ApplyTransforms
from nipype.pipeline import engine as pe
from pathlib import Path

def rsfmri_preproc():


    #connect(source, "source_output", dest, "dest_input")
    #connect([(source, dest, [("source_output1", "dest_input1"),
    #                     ("source_output2", "dest_input2")
    #                     ])
    #     ])
    wf = Workflow(name="mouce_rsfmri", base_dir="/output/working_dir")

    # Anatomical workflow:
    # 1. Bias field correction
    # 2. Skull stripping and brain extraction
    # 3. Spatial normalization
    # 4. Brain tissue segmentation

    # 1. Bias field correction


    # 2. Skull stripping and brain extraction
    brain_extraction_wf = init_rodent_brain_extraction_wf(
        template_id=skull_strip_template.space,
        omp_nthreads=omp_nthreads,
        debug=debug,
        mri_scheme="T1w",
        ses=ses
    )

    # 3. Spatial normalization and registration
    anat_norm_wf = init_anat_norm_wf(
        debug=debug,
        omp_nthreads=omp_nthreads,
        templates=spaces.get_spaces(nonstandard=False, dim=(3,)),
    )

    # 4. Brain tissue segmentation
    MouseIn=True
    if MouseIn:
        gm_tpm = Path("/globalscratch/users/q/d/qdessain/SYRINA/Template/MouseIn/tpl-MouseIn_res-1_desc-GM_probseg.nii.gz")#get("MouseIn", label="GM", suffix="probseg")
        wm_tpm = Path("/globalscratch/users/q/d/qdessain/SYRINA/Template/MouseIn/tpl-MouseIn_res-1_desc-WM_probseg.nii.gz")#get("MouseIn", label="WM", suffix="probseg")
        csf_tpm = Path("/globalscratch/users/q/d/qdessain/SYRINA/Template/MouseIn/tpl-MouseIn_res-1_desc-CSF_probseg.nii.gz")#get("MouseIn", label="CSF", suffix="probseg")
    else:
        gm_tpm = Path("/globalscratch/users/q/d/qdessain/SYRINA/Template/TMBTA/tpl-TMBTA_gm.nii.gz")#get("MouseIn", label="GM", suffix="probseg")
        wm_tpm = Path("/globalscratch/users/q/d/qdessain/SYRINA/Template/TMBTA/tpl-TMBTA_wm.nii.gz")#get("MouseIn", label="WM", suffix="probseg")
        csf_tpm = Path("/globalscratch/users/q/d/qdessain/SYRINA/Template/TMBTA/tpl-TMBTA_csf.nii.gz")#get("MouseIn", label="CSF", suffix="probseg")

    xfm_gm = pe.Node(
        ApplyTransforms(input_image=_pop(gm_tpm), interpolation="MultiLabel"),
        name="xfm_gm",
    )
    xfm_wm = pe.Node(
        ApplyTransforms(input_image=_pop(wm_tpm), interpolation="MultiLabel"),
        name="xfm_wm",
    )
    xfm_csf = pe.Node(
        ApplyTransforms(input_image=_pop(csf_tpm), interpolation="MultiLabel"),
        name="xfm_csf",
    )


    # Functional workflow:

    # 1. Bias field correction
    # 2. Slice timing correction
    # 3. Motion correction
    # 4. Registration to anatomical space
    # 5. Registration to template space
    # 6. Spatial normalization
    # 7. Brain tissue segmentation
    # 8. Confound extraction
    # 9. Denoising
    # 10. Smoothing

    

def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist



def init_anat_norm_wf(
    *, debug, omp_nthreads, templates, name="anat_norm_wf",
):
    """
    Build an individual spatial normalization workflow using ``antsRegistration``.

    Workflow Graph
        .. workflow ::
            :graph2use: orig
            :simple_form: yes

            from fprodents.patch.workflows.anatomical import init_anat_norm_wf
            wf = init_anat_norm_wf(
                debug=False,
                omp_nthreads=1,
                templates=['Fischer344'],
            )

    .. important::
        This workflow defines an iterable input over the input parameter ``templates``,
        so Nipype will produce one copy of the downstream workflows which connect
        ``poutputnode.template`` or ``poutputnode.template_spec`` to their inputs
        (``poutputnode`` stands for *parametric output node*).
        Nipype refers to this expansion of the graph as *parameterized execution*.
        If a joint list of values is required (and thus cutting off parameterization),
        please use the equivalent outputs of ``outputnode`` (which *joins* all the
        parameterized execution paths).

    Parameters
    ----------
    debug : :obj:`bool`
        Apply sloppy arguments to speed up processing. Use with caution,
        registration processes will be very inaccurate.
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use.
    templates : :obj:`list` of :obj:`str`
        List of standard space fullnames (e.g., ``MNI152NLin6Asym``
        or ``MNIPediatricAsym:cohort-4``) which are targets for spatial
        normalization.

    Inputs
    ------
    moving_image
        The input image that will be normalized to standard space.
    moving_mask
        A precise brain mask separating skull/skin/fat from brain
        structures.
    lesion_mask
        (optional) A mask to exclude regions from the cost-function
        input domain to enable standardization of lesioned brains.
    orig_t1w
        The original T1w image from the BIDS structure.
    template
        Template name and specification

    Outputs
    -------
    standardized
        The T1w after spatial normalization, in template space.
    anat2std_xfm
        The T1w-to-template transform.
    std2anat_xfm
        The template-to-T1w transform.
    std_mask
        The ``moving_mask`` in template space (matches ``standardized`` output).
    template
        Template name extracted from the input parameter ``template``, for further
        use in downstream nodes.
    template_spec
        Template specifications extracted from the input parameter ``template``, for
        further use in downstream nodes.

    """
    from collections import defaultdict
    from nipype.interfaces.ants import ImageMath
    from ..interfaces import RobustMNINormalization

    ntpls = len(templates)
    workflow = Workflow(name=name)

    inputnode = pe.Node(
        niu.IdentityInterface(
            fields=[
                "lesion_mask",
                "moving_image",
                "moving_mask",
                "orig_t1w",
                "template",
            ]
        ),
        name="inputnode",
    )
    inputnode.iterables = [("template", templates)]

    out_fields = [
        "anat2std_xfm",
        "standardized",
        "std2anat_xfm",
        "std_mask",
        "template",
        "template_spec",
    ]
    poutputnode = pe.Node(niu.IdentityInterface(fields=out_fields), name="poutputnode")



    # With the improvements from nipreps/niworkflows#342 this truncation is now necessary
    trunc_mov = pe.Node(
        ImageMath(operation="TruncateImageIntensity", op2="0.01 0.999 256"),
        name="trunc_mov",
    )

    registration = pe.Node(
        RobustMNINormalization(float=True, flavor=["precise", "testing"][debug],),
        name="registration",
        n_procs=omp_nthreads,
        mem_gb=2,
    )

    tpl_moving = pe.Node(
        ApplyTransforms(
            dimension=3,
            default_value=0,
            interpolation="LanczosWindowedSinc",
        ),
        name="tpl_moving",
    )

    std_mask = pe.Node(ApplyTransforms(interpolation="MultiLabel"), name="std_mask")

    MouseIn=True
    if MouseIn:
        tpl_moving.inputs.reference_image = "/globalscratch/users/q/d/qdessain/SYRINA/Template/MouseIn/tpl-MouseIn_res-1_T1map.nii.gz"  # before
        std_mask.inputs.reference_image = "/globalscratch/users/q/d/qdessain/SYRINA/Template/MouseIn/tpl-MouseIn_res-1_desc-brain_mask.nii.gz"
    else:
        tpl_moving.inputs.reference_image = "/globalscratch/users/q/d/qdessain/SYRINA/Template/TMBTA/tpl-TMBTA_T1wBrain.nii.gz"  # before
        std_mask.inputs.reference_image = "/globalscratch/users/q/d/qdessain/SYRINA/Template/TMBTA/tpl-TMBTA_desc-brain_mask.nii.gz"
    # fmt:off
    workflow.connect([
        (inputnode, poutputnode, [('template', 'template')]),
        (inputnode, trunc_mov, [('moving_image', 'op1')]),
        (inputnode, registration, [
            ('moving_mask', 'moving_mask'),
            ('lesion_mask', 'lesion_mask')]),
        (inputnode, tpl_moving, [('moving_image', 'input_image')]),
        (inputnode, std_mask, [('moving_mask', 'input_image')]),
        (split_desc, registration, [('name', 'template'),
                                    (('spec', _no_atlas), 'template_spec')]),
        (trunc_mov, registration, [
            ('output_image', 'moving_image')]),
        (registration, tpl_moving, [('composite_transform', 'transforms')]),
        (registration, std_mask, [('composite_transform', 'transforms')]),
        (registration, poutputnode, [
            ('composite_transform', 'anat2std_xfm'),
            ('inverse_composite_transform', 'std2anat_xfm')]),
        (tpl_moving, poutputnode, [('output_image', 'standardized')]),
        (std_mask, poutputnode, [('output_image', 'std_mask')]),
    ])
    # fmt:on

    # Provide synchronized output
    outputnode = pe.JoinNode(
        niu.IdentityInterface(fields=out_fields),
        name="outputnode",
        joinsource="inputnode",
    )
    # fmt:off
    workflow.connect([
        (poutputnode, outputnode, [(f, f) for f in out_fields]),
    ])
    # fmt:on

    return workflow

def _no_atlas(spec):
    spec["atlas"] = None
    return spec
