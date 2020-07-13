"""Patched workflows for compatibility"""

from nipype import logging
from nipype.pipeline import engine as pe
from nipype.interfaces import fsl, utility as niu

from nipype.interfaces.ants.base import Info as ANTsInfo

from niworkflows.interfaces.images import ValidateImage
from niworkflows.engine.workflows import LiterateWorkflow as Workflow
from niworkflows.interfaces.utility import KeySelect

from smriprep.utils.misc import apply_lut as _apply_bids_lut
from smriprep.workflows.anat import init_anat_template_wf
from smriprep.workflows.norm import init_anat_norm_wf
from smriprep.workflows.outputs import init_anat_reports_wf, init_anat_derivatives_wf

from nirodents.workflows.brainextraction import init_rodent_brain_extraction_wf

from ..utils import fix_multi_source_name

LOGGER = logging.getLogger('nipype.workflow')


def init_anat_preproc_wf(
        *,
        bids_root,
        hires,
        longitudinal,
        t2w,
        omp_nthreads,
        output_dir,
        skull_strip_mode,
        skull_strip_template,
        spaces,
        debug=False,
        existing_derivatives=None,
        freesurfer=False,
        name='anat_preproc_wf',
        skull_strip_fixed_seed=False,
):
    """
    Stage the anatomical preprocessing steps of *sMRIPrep*.
    This includes:
      - T1w reference: realigning and then averaging T1w images.
      - Brain extraction and INU (bias field) correction.
      - Brain tissue segmentation.
      - Spatial normalization to standard spaces.
      - Surface reconstruction with FreeSurfer_.
    .. include:: ../links.rst
    Workflow Graph
        .. workflow::
            :graph2use: orig
            :simple_form: yes
            from niworkflows.utils.spaces import SpatialReferences, Reference
            from smriprep.workflows.anatomical import init_anat_preproc_wf
            wf = init_anat_preproc_wf(
                bids_root='.',
                freesurfer=True,
                hires=True,
                longitudinal=False,
                t1w=['t1w.nii.gz'],
                omp_nthreads=1,
                output_dir='.',
                skull_strip_mode='force',
                skull_strip_template=Reference('OASIS30ANTs'),
                spaces=SpatialReferences(spaces=['MNI152NLin2009cAsym', 'fsaverage5']),
            )
    Parameters
    ----------
    bids_root : :obj:`str`
        Path of the input BIDS dataset root
    existing_derivatives : :obj:`dict` or None
        Dictionary mapping output specification attribute names and
        paths to corresponding derivatives.
    freesurfer : :obj:`bool`
        Enable FreeSurfer surface reconstruction (increases runtime by 6h,
        at the very least)
    hires : :obj:`bool`
        Enable sub-millimeter preprocessing in FreeSurfer
    longitudinal : :obj:`bool`
        Create unbiased structural template, regardless of number of inputs
        (may increase runtime)
    t1w : :obj:`list`
        List of T1-weighted structural images.
    omp_nthreads : :obj:`int`
        Maximum number of threads an individual process may use
    output_dir : :obj:`str`
        Directory in which to save derivatives
    skull_strip_template : :py:class:`~niworkflows.utils.spaces.Reference`
        Spatial reference to use in atlas-based brain extraction.
    spaces : :py:class:`~niworkflows.utils.spaces.SpatialReferences`
        Object containing standard and nonstandard space specifications.
    debug : :obj:`bool`
        Enable debugging outputs
    name : :obj:`str`, optional
        Workflow name (default: anat_preproc_wf)
    skull_strip_mode : :obj:`str`
        Determiner for T1-weighted skull stripping (`force` ensures skull stripping,
        `skip` ignores skull stripping, and `auto` automatically ignores skull stripping
        if pre-stripped brains are detected).
    skull_strip_fixed_seed : :obj:`bool`
        Do not use a random seed for skull-stripping - will ensure
        run-to-run replicability when used with --omp-nthreads 1
        (default: ``False``).
    Inputs
    ------
    t1w
        List of T1-weighted structural images
    t2w
        List of T2-weighted structural images
    roi
        A mask to exclude regions during standardization
    flair
        List of FLAIR images
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    subject_id
        FreeSurfer subject ID
    Outputs
    -------
    t1w_preproc
        The T1w reference map, which is calculated as the average of bias-corrected
        and preprocessed T1w images, defining the anatomical space.
    t1w_brain
        Skull-stripped ``t1w_preproc``
    t1w_mask
        Brain (binary) mask estimated by brain extraction.
    t1w_dseg
        Brain tissue segmentation of the preprocessed structural image, including
        gray-matter (GM), white-matter (WM) and cerebrospinal fluid (CSF).
    t1w_tpms
        List of tissue probability maps corresponding to ``t1w_dseg``.
    std_preproc
        T1w reference resampled in one or more standard spaces.
    std_mask
        Mask of skull-stripped template, in MNI space
    std_dseg
        Segmentation, resampled into MNI space
    std_tpms
        List of tissue probability maps in MNI space
    subjects_dir
        FreeSurfer SUBJECTS_DIR
    anat2std_xfm
        Nonlinear spatial transform to resample imaging data given in anatomical space
        into standard space.
    std2anat_xfm
        Inverse transform of the above.
    subject_id
        FreeSurfer subject ID
    t1w2fsnative_xfm
        LTA-style affine matrix translating from T1w to
        FreeSurfer-conformed subject space
    fsnative2t1w_xfm
        LTA-style affine matrix translating from FreeSurfer-conformed
        subject space to T1w
    surfaces
        GIFTI surfaces (gray/white boundary, midthickness, pial, inflated)
    See Also
    --------
    * :py:func:`~niworkflows.anat.ants.init_brain_extraction_wf`
    * :py:func:`~smriprep.workflows.surfaces.init_surface_recon_wf`
    """
    freesurfer = False
    workflow = Workflow(name=name)
    num_t2w = len(t2w)
    desc = """Anatomical data preprocessing
: """
    desc += """\
A total of {num_t2w} T2-weighted (T2w) images were found within the input
BIDS dataset."""

    inputnode = pe.Node(
        niu.IdentityInterface(fields=['t2w', 'roi', 'flair', 'subjects_dir', 'subject_id']),
        name='inputnode')

    outputnode = pe.Node(niu.IdentityInterface(
        fields=['t2w_preproc', 't2w_mask', 't2w_dseg', 't2w_tpms',
                'std_preproc', 'std_mask', 'std_dseg', 'std_tpms',
                'anat2std_xfm', 'std2anat_xfm', 'template']),
        name='outputnode')

    # Connect reportlets workflows
    anat_reports_wf = init_anat_reports_wf(
        freesurfer=freesurfer,
        output_dir=output_dir,
    )
    workflow.connect([
        (outputnode, anat_reports_wf, [
            ('t2w_preproc', 'inputnode.t1w_preproc'),
            ('t2w_mask', 'inputnode.t1w_mask'),
            ('t2w_dseg', 'inputnode.t1w_dseg')]),
    ])

    if existing_derivatives is not None:
        LOGGER.log(25, "Anatomical workflow will reuse prior derivatives found in the "
                   "output folder (%s).", output_dir)
        desc += """
Anatomical preprocessing was reused from previously existing derivative objects.\n"""
        workflow.__desc__ = desc

        templates = existing_derivatives.pop('template')
        templatesource = pe.Node(niu.IdentityInterface(
            fields=['template']), name='templatesource')
        templatesource.iterables = [('template', templates)]
        outputnode.inputs.template = templates

        for field, value in existing_derivatives.items():
            setattr(outputnode.inputs, field, value)

        anat_reports_wf.inputs.inputnode.source_file = fix_multi_source_name(
            [existing_derivatives['t2w_preproc']], modality='T2w')

        stdselect = pe.Node(KeySelect(
            fields=['std_preproc', 'std_mask'], keys=templates),
            name='stdselect', run_without_submitting=True)
        workflow.connect([
            (inputnode, outputnode, [('subjects_dir', 'subjects_dir'),
                                     ('subject_id', 'subject_id')]),
            (inputnode, anat_reports_wf, [
                ('subjects_dir', 'inputnode.subjects_dir'),
                ('subject_id', 'inputnode.subject_id')]),
            (templatesource, stdselect, [('template', 'key')]),
            (outputnode, stdselect, [('std_preproc', 'std_preproc'),
                                     ('std_mask', 'std_mask')]),
            (stdselect, anat_reports_wf, [
                ('key', 'inputnode.template'),
                ('std_preproc', 'inputnode.std_t1w'),
                ('std_mask', 'inputnode.std_mask'),
            ]),
        ])
        return workflow

    # The workflow is not cached.
    desc += """
All of them were corrected for intensity non-uniformity (INU)
""" if num_t2w > 1 else """\
The T2-weighted (T2w) image was corrected for intensity non-uniformity (INU)
"""
    desc += """\
with `N4BiasFieldCorrection` [@n4], distributed with ANTs {ants_ver} \
[@ants, RRID:SCR_004757]"""
    desc += '.\n' if num_t2w > 1 else ", and used as T2w-reference throughout the workflow.\n"

    desc += """\
The T2w-reference was then skull-stripped with a *Nipype* implementation of
the `antsBrainExtraction.sh` workflow (from ANTs), using {skullstrip_tpl}
as target template.
Brain tissue segmentation of cerebrospinal fluid (CSF),
white-matter (WM) and gray-matter (GM) was performed on
the brain-extracted T1w using `fast` [FSL {fsl_ver}, RRID:SCR_002823,
@fsl_fast].
"""

    workflow.__desc__ = desc.format(
        ants_ver=ANTsInfo.version() or '(version unknown)',
        fsl_ver=fsl.FAST().version or '(version unknown)',
        num_t2w=num_t2w,
        skullstrip_tpl=skull_strip_template.fullname,
    )

    buffernode = pe.Node(niu.IdentityInterface(
        fields=['t2w_brain', 't2w_mask']), name='buffernode')

    # 1. Anatomical reference generation - average input T1w images.
    anat_template_wf = init_anat_template_wf(longitudinal=longitudinal, omp_nthreads=omp_nthreads,
                                             num_t1w=num_t2w)

    anat_validate = pe.Node(ValidateImage(), name='anat_validate',
                            run_without_submitting=True)

    # 2. Brain-extraction and INU (bias field) correction.
    if skull_strip_mode == 'auto':
        import numpy as np
        import nibabel as nb

        def _is_skull_stripped(imgs):
            """Check if T1w images are skull-stripped."""
            def _check_img(img):
                data = np.abs(nb.load(img).get_fdata(dtype=np.float32))
                sidevals = data[0, :, :].sum() + data[-1, :, :].sum() + \
                    data[:, 0, :].sum() + data[:, -1, :].sum() + \
                    data[:, :, 0].sum() + data[:, :, -1].sum()
                return sidevals < 10

            return all(_check_img(img) for img in imgs)

        skull_strip_mode = _is_skull_stripped(t2w)

    if skull_strip_mode in (True, 'skip'):
        raise NotImplementedError("Cannot run on already skull-stripped images.")
    else:
        # ants_affine_init?
        brain_extraction_wf = init_rodent_brain_extraction_wf(
            in_template=skull_strip_template.space,
            omp_nthreads=omp_nthreads,
        )

    # 4. Spatial normalization
    anat_norm_wf = init_anat_norm_wf(
        debug=debug,
        omp_nthreads=omp_nthreads,
        templates=spaces.get_spaces(nonstandard=False, dim=(3,)),
    )

    workflow.connect([
        # Step 1.
        (inputnode, anat_template_wf, [('t2w', 'inputnode.t1w')]),
        (anat_template_wf, anat_validate, [
            ('outputnode.t1w_ref', 'in_file')]),
        (anat_validate, brain_extraction_wf, [
            ('out_file', 'inputnode.in_files')]),
        (brain_extraction_wf, outputnode, [
            (('outputnode.out_corrected', _pop), 't2w_preproc')]),
        (anat_template_wf, outputnode, [
            ('outputnode.t1w_realign_xfm', 't2w_ref_xfms')]),
        (buffernode, outputnode, [('t2w_brain', 't2w_brain'),
                                  ('t2w_mask', 't2w_mask')]),
        # Steps 2, 3 and 4
        (inputnode, anat_norm_wf, [
            (('t2w', fix_multi_source_name), 'inputnode.orig_t1w'),
            ('roi', 'inputnode.lesion_mask')]),
        (brain_extraction_wf, anat_norm_wf, [
            (('outputnode.out_corrected', _pop), 'inputnode.moving_image')]),
        (buffernode, anat_norm_wf, [('t2w_mask', 'inputnode.moving_mask')]),
        (anat_norm_wf, outputnode, [
            ('poutputnode.standardized', 'std_preproc'),
            ('poutputnode.std_mask', 'std_mask'),
            ('poutputnode.std_dseg', 'std_dseg'),
            ('poutputnode.std_tpms', 'std_tpms'),
            ('outputnode.template', 'template'),
            ('outputnode.anat2std_xfm', 'anat2std_xfm'),
            ('outputnode.std2anat_xfm', 'std2anat_xfm'),
        ]),
    ])

    # Change LookUp Table - BIDS wants: 0 (bg), 1 (gm), 2 (wm), 3 (csf)
    lut_t1w_dseg = pe.Node(niu.Function(function=_apply_bids_lut),
                           name='lut_t1w_dseg')

    workflow.connect([
        (lut_t1w_dseg, anat_norm_wf, [
            ('out', 'inputnode.moving_segmentation')]),
        (lut_t1w_dseg, outputnode, [('out', 't1w_dseg')]),
    ])

    # Connect reportlets
    workflow.connect([
        (inputnode, anat_reports_wf, [
            (('t2w', fix_multi_source_name), 'inputnode.source_file')]),
        (outputnode, anat_reports_wf, [
            ('std_preproc', 'inputnode.std_t1w'),
            ('std_mask', 'inputnode.std_mask'),
        ]),
        (anat_template_wf, anat_reports_wf, [
            ('outputnode.out_report', 'inputnode.t1w_conform_report')]),
        (anat_norm_wf, anat_reports_wf, [
            ('poutputnode.template', 'inputnode.template')]),
    ])

    # Write outputs ############################################3
    anat_derivatives_wf = init_anat_derivatives_wf(
        bids_root=bids_root,
        freesurfer=freesurfer,
        num_t1w=num_t2w,
        output_dir=output_dir,
    )

    workflow.connect([
        # Connect derivatives
        (anat_template_wf, anat_derivatives_wf, [
            ('outputnode.t1w_valid_list', 'inputnode.source_files')]),
        (anat_norm_wf, anat_derivatives_wf, [
            ('poutputnode.template', 'inputnode.template'),
            ('poutputnode.anat2std_xfm', 'inputnode.anat2std_xfm'),
            ('poutputnode.std2anat_xfm', 'inputnode.std2anat_xfm')
        ]),
        (outputnode, anat_derivatives_wf, [
            ('std_preproc', 'inputnode.std_t1w'),
            ('t2w_ref_xfms', 'inputnode.t1w_ref_xfms'),
            ('t2w_preproc', 'inputnode.t1w_preproc'),
            ('t2w_mask', 'inputnode.t1w_mask'),
            ('t2w_dseg', 'inputnode.t1w_dseg'),
            ('t2w_tpms', 'inputnode.t1w_tpms'),
            ('std_mask', 'inputnode.std_mask'),
            ('std_dseg', 'inputnode.std_dseg'),
            ('std_tpms', 'inputnode.std_tpms'),
        ]),
    ])

    if not freesurfer:  # Flag --fs-no-reconall is set - return
        # Brain tissue segmentation - FAST produces: 0 (bg), 1 (wm), 2 (csf), 3 (gm)
        t1w_dseg = pe.Node(fsl.FAST(segments=True, no_bias=True, probability_maps=True),
                           name='t1w_dseg', mem_gb=3)
        lut_t1w_dseg.inputs.lut = (0, 3, 1, 2)  # Maps: 0 -> 0, 3 -> 1, 1 -> 2, 2 -> 3.
        fast2bids = pe.Node(niu.Function(function=_probseg_fast2bids), name="fast2bids",
                            run_without_submitting=True)

        workflow.connect([
            (brain_extraction_wf, buffernode, [
                (('outputnode.out_brain', _pop), 't2w_brain'),
                ('outputnode.out_mask', 't2w_mask')]),
            (buffernode, t1w_dseg, [('t2w_brain', 'in_files')]),
            (t1w_dseg, lut_t1w_dseg, [('partial_volume_map', 'in_dseg')]),
            (t1w_dseg, fast2bids, [('partial_volume_files', 'inlist')]),
            (fast2bids, anat_norm_wf, [('out', 'inputnode.moving_tpms')]),
            (fast2bids, outputnode, [('out', 't2w_tpms')]),
        ])
        return workflow


def _pop(inlist):
    if isinstance(inlist, (list, tuple)):
        return inlist[0]
    return inlist


def _probseg_fast2bids(inlist):
    """Reorder a list of probseg maps from FAST (CSF, WM, GM) to BIDS (GM, WM, CSF)."""
    return (inlist[1], inlist[2], inlist[0])
