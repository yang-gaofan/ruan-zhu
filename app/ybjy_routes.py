import os
from pathlib import Path

from flask import flash, redirect, render_template, request, send_from_directory, url_for

from ybjy_config import APP_SHORT_NAME, APP_TITLE, EXPORT_DIR, TOP_K_DEFAULT, UPLOAD_DIR
from app.ybjy_business_service import (
    build_ybjy_function_description_text,
    build_ybjy_storage_name,
    export_ybjy_feedback_csv,
    export_ybjy_project_csv,
    is_allowed_ybjy_image,
    run_ybjy_image_search,
    run_ybjy_text_search,
)
from app.ybjy_clip_service import frozen_ybjy_clip_service
from app.ybjy_database import (
    create_ybjy_project,
    delete_ybjy_project,
    delete_ybjy_sample,
    get_ybjy_project,
    get_ybjy_sample,
    list_ybjy_projects,
    list_ybjy_samples_by_project,
    save_ybjy_feedback,
    save_ybjy_sample,
    update_ybjy_project,
)
from app.ybjy_project_service import (
    build_ybjy_operation_checklist,
    build_ybjy_project_overview,
    build_ybjy_project_report_text,
    export_ybjy_project_inventory_csv,
)
from app.ybjy_sample_inspection_service import (
    build_ybjy_inspection_page_data,
    export_ybjy_sample_inspection_csv,
)
from app.ybjy_representation_service import (
    build_ybjy_representation_lab_data,
    export_ybjy_representation_experiments_csv,
    run_ybjy_projection_head_training,
)


def _read_ybjy_positive_int(form_name, default_value, max_value=50):
    raw_value = request.form.get(form_name, str(default_value)).strip()
    try:
        number_value = int(raw_value)
    except ValueError:
        return default_value
    return max(1, min(number_value, max_value))


def _read_ybjy_float(form_name, default_value, min_value, max_value):
    raw_value = request.form.get(form_name, str(default_value)).strip()
    try:
        number_value = float(raw_value)
    except ValueError:
        return default_value
    return max(min_value, min(number_value, max_value))


def register_yangben_yujian_routes(app):
    @app.context_processor
    def inject_app_title():
        return {
            'app_title': APP_TITLE,
            'app_short_name': APP_SHORT_NAME,
        }

    @app.route('/')
    def ybjy_home():
        project_rows = list_ybjy_projects()
        total_project_count = len(project_rows)
        total_sample_count = sum(row.get('sample_count', 0) for row in project_rows)
        return render_template(
            'ybjy_home.html',
            project_rows=project_rows,
            total_project_count=total_project_count,
            total_sample_count=total_sample_count,
        )

    @app.route('/help')
    def ybjy_help():
        return render_template('ybjy_help.html')

    @app.route('/project/create', methods=['POST'])
    def ybjy_project_create():
        project_name = request.form.get('project_name', '').strip()
        project_note = request.form.get('project_note', '').strip()
        if not project_name:
            flash('项目名称不能为空。')
            return redirect(url_for('ybjy_home'))
        project_id = create_ybjy_project(project_name, project_note)
        flash('项目创建成功，可以开始导入图片样本。')
        return redirect(url_for('ybjy_project_detail', project_id=project_id))

    @app.route('/project/<int:project_id>/update', methods=['POST'])
    def ybjy_project_update(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))
        project_name = request.form.get('project_name', '').strip()
        project_note = request.form.get('project_note', '').strip()
        if not project_name:
            flash('项目名称不能为空。')
            return redirect(url_for('ybjy_home'))
        update_ybjy_project(project_id, project_name, project_note)
        flash('项目信息已更新。')
        return redirect(url_for('ybjy_home'))

    @app.route('/project/<int:project_id>/delete', methods=['POST'])
    def ybjy_project_delete(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))
        delete_ybjy_project(project_id)
        flash('项目已删除。')
        return redirect(url_for('ybjy_home'))

    @app.route('/project/<int:project_id>')
    def ybjy_project_detail(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('未找到对应项目。')
            return redirect(url_for('ybjy_home'))
        sample_rows = list_ybjy_samples_by_project(project_id)
        return render_template(
            'ybjy_project_detail.html',
            project_info=project_info,
            sample_rows=sample_rows,
            sample_count=len(sample_rows),
        )

    @app.route('/project/<int:project_id>/upload', methods=['POST'])
    def ybjy_upload_sample(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))

        uploaded_files = request.files.getlist('sample_files')
        if not uploaded_files or uploaded_files[0].filename == '':
            flash('请先选择图片文件。')
            return redirect(url_for('ybjy_project_detail', project_id=project_id))

        current_count = len(list_ybjy_samples_by_project(project_id))
        success_count = 0
        ignored_count = 0

        for file_item in uploaded_files:
            if not file_item or not file_item.filename:
                ignored_count += 1
                continue
            if not is_allowed_ybjy_image(file_item.filename):
                ignored_count += 1
                continue

            current_count += 1
            storage_name = build_ybjy_storage_name(file_item.filename, current_count)
            project_folder = Path(UPLOAD_DIR) / f'project_{project_id}'
            os.makedirs(project_folder, exist_ok=True)
            file_path = project_folder / storage_name
            file_item.save(file_path)

            image_feature = frozen_ybjy_clip_service.encode_image_to_feature(str(file_path))
            relative_path = f'project_{project_id}/{storage_name}'
            save_ybjy_sample(project_id, file_item.filename, relative_path, image_feature)
            success_count += 1

        if success_count:
            flash(f'成功导入 {success_count} 张图片，并完成特征提取。')
        if ignored_count:
            flash(f'有 {ignored_count} 个文件未导入，请确认格式为 JPG、PNG、BMP 或 WEBP。')
        return redirect(url_for('ybjy_project_detail', project_id=project_id))

    @app.route('/sample/<int:sample_id>/delete', methods=['POST'])
    def ybjy_delete_sample(sample_id):
        sample_info = get_ybjy_sample(sample_id)
        if not sample_info:
            flash('样本不存在。')
            return redirect(url_for('ybjy_home'))
        project_id = sample_info['project_id']
        delete_ybjy_sample(sample_id)
        flash('样本删除成功。')
        return redirect(url_for('ybjy_project_detail', project_id=project_id))

    @app.route('/project/<int:project_id>/text-search', methods=['GET', 'POST'])
    def ybjy_text_search(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))

        result_rows = []
        query_text = ''
        top_k = TOP_K_DEFAULT

        if request.method == 'POST':
            query_text = request.form.get('query_text', '').strip()
            top_k = _read_ybjy_positive_int('top_k', TOP_K_DEFAULT)
            if not query_text:
                flash('请输入检索文字。')
            else:
                text_feature = frozen_ybjy_clip_service.encode_text_to_feature(query_text)
                result_rows = run_ybjy_text_search(project_id, query_text, text_feature, top_k)

        return render_template(
            'ybjy_text_search.html',
            project_info=project_info,
            result_rows=result_rows,
            query_text=query_text,
            top_k=top_k,
        )

    @app.route('/project/<int:project_id>/image-search', methods=['GET', 'POST'])
    def ybjy_image_search(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))

        sample_rows = list_ybjy_samples_by_project(project_id)
        result_rows = []
        source_sample_id = 0
        top_k = TOP_K_DEFAULT

        if request.method == 'POST':
            source_sample_id = int(request.form.get('source_sample_id', '0'))
            top_k = _read_ybjy_positive_int('top_k', TOP_K_DEFAULT)
            if source_sample_id <= 0:
                flash('请选择查询图片。')
            else:
                result_rows = run_ybjy_image_search(project_id, source_sample_id, top_k)

        return render_template(
            'ybjy_image_search.html',
            project_info=project_info,
            sample_rows=sample_rows,
            result_rows=result_rows,
            source_sample_id=source_sample_id,
            top_k=top_k,
        )

    @app.route('/project/<int:project_id>/feedback', methods=['POST'])
    def ybjy_feedback_save(project_id):
        target_sample_id = int(request.form.get('target_sample_id', '0'))
        query_type = request.form.get('query_type', '').strip()
        query_value = request.form.get('query_value', '').strip()
        feedback_value = int(request.form.get('feedback_value', '0'))

        if target_sample_id <= 0 or query_type == '' or query_value == '':
            flash('反馈参数不完整。')
            return redirect(url_for('ybjy_project_detail', project_id=project_id))

        save_ybjy_feedback(project_id, target_sample_id, query_type, query_value, feedback_value)
        flash('人工校正记录已保存，再次执行同一检索时会参与结果排序。')
        return redirect(request.referrer or url_for('ybjy_project_detail', project_id=project_id))

    @app.route('/project/<int:project_id>/export-center')
    def ybjy_export_center(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))
        sample_rows = list_ybjy_samples_by_project(project_id)
        return render_template(
            'ybjy_export_center.html',
            project_info=project_info,
            sample_rows=sample_rows,
        )

    @app.route('/project/<int:project_id>/report')
    def ybjy_project_report(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))

        overview = build_ybjy_project_overview(project_info)
        checklist_rows = build_ybjy_operation_checklist(project_info)
        report_text = build_ybjy_project_report_text(project_info)
        return render_template(
            'ybjy_project_report.html',
            project_info=project_info,
            overview=overview,
            checklist_rows=checklist_rows,
            report_text=report_text,
        )

    @app.route('/project/<int:project_id>/export-inventory')
    def ybjy_export_inventory(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))

        export_name = export_ybjy_project_inventory_csv(project_info)
        return send_from_directory(EXPORT_DIR, export_name, as_attachment=True)

    @app.route('/project/<int:project_id>/sample-inspection')
    def ybjy_sample_inspection(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))

        page_data = build_ybjy_inspection_page_data(project_info)
        return render_template('ybjy_sample_inspection.html', **page_data)

    @app.route('/project/<int:project_id>/export-sample-inspection')
    def ybjy_export_sample_inspection(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))

        export_name = export_ybjy_sample_inspection_csv(project_info)
        return send_from_directory(EXPORT_DIR, export_name, as_attachment=True)

    @app.route('/project/<int:project_id>/representation-lab', methods=['GET', 'POST'])
    def ybjy_representation_lab(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))

        queue_size = _read_ybjy_positive_int('queue_size', 64, max_value=4096)
        if request.method == 'POST':
            train_options = {
                'experiment_name': request.form.get('experiment_name', '').strip() or '图像特征对比训练记录',
                'queue_size': queue_size,
                'projection_dim': _read_ybjy_positive_int('projection_dim', 128, max_value=1024),
                'epoch_count': _read_ybjy_positive_int('epoch_count', 12, max_value=100),
                'learning_rate': _read_ybjy_float('learning_rate', 0.001, 0.00001, 0.1),
                'temperature_value': _read_ybjy_float('temperature_value', 0.2, 0.03, 1.0),
                'noise_scale': _read_ybjy_float('noise_scale', 0.03, 0.0, 0.2),
            }
            train_result = run_ybjy_projection_head_training(project_id, train_options)
            flash(train_result['message'])
            return redirect(url_for('ybjy_representation_lab', project_id=project_id))

        page_data = build_ybjy_representation_lab_data(project_info, queue_size)
        return render_template('ybjy_representation_lab.html', **page_data)

    @app.route('/project/<int:project_id>/export-representation-experiments')
    def ybjy_export_representation_experiments(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))

        export_name = export_ybjy_representation_experiments_csv(project_info)
        return send_from_directory(EXPORT_DIR, export_name, as_attachment=True)

    @app.route('/project/<int:project_id>/export-text-result', methods=['POST'])
    def ybjy_export_text_result(project_id):
        project_info = get_ybjy_project(project_id)
        query_text = request.form.get('query_text', '').strip()
        if not project_info or not query_text:
            flash('请先输入要导出的检索文字。')
            return redirect(url_for('ybjy_export_center', project_id=project_id))

        text_feature = frozen_ybjy_clip_service.encode_text_to_feature(query_text)
        result_rows = run_ybjy_text_search(project_id, query_text, text_feature, TOP_K_DEFAULT)
        export_name = export_ybjy_project_csv(project_info, result_rows, 'text_search')
        return send_from_directory(EXPORT_DIR, export_name, as_attachment=True)

    @app.route('/project/<int:project_id>/export-image-result', methods=['POST'])
    def ybjy_export_image_result(project_id):
        project_info = get_ybjy_project(project_id)
        source_sample_id = int(request.form.get('source_sample_id', '0'))
        if not project_info or source_sample_id <= 0:
            flash('请先选择要导出的查询图片。')
            return redirect(url_for('ybjy_export_center', project_id=project_id))

        result_rows = run_ybjy_image_search(project_id, source_sample_id, TOP_K_DEFAULT)
        export_name = export_ybjy_project_csv(project_info, result_rows, 'image_search')
        return send_from_directory(EXPORT_DIR, export_name, as_attachment=True)

    @app.route('/project/<int:project_id>/export-feedback')
    def ybjy_export_feedback(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))
        export_name = export_ybjy_feedback_csv(project_info)
        return send_from_directory(EXPORT_DIR, export_name, as_attachment=True)

    @app.route('/project/<int:project_id>/function-description')
    def ybjy_function_description(project_id):
        project_info = get_ybjy_project(project_id)
        if not project_info:
            flash('项目不存在。')
            return redirect(url_for('ybjy_home'))
        function_text = build_ybjy_function_description_text()
        return render_template(
            'ybjy_function_description.html',
            project_info=project_info,
            function_text=function_text,
        )

    @app.route('/uploads/<path:relative_path>')
    def ybjy_uploaded_file(relative_path):
        folder_name = os.path.dirname(relative_path)
        file_name = os.path.basename(relative_path)
        return send_from_directory(Path(UPLOAD_DIR) / folder_name, file_name)

    @app.route('/exports/<path:file_name>')
    def ybjy_exported_file(file_name):
        return send_from_directory(EXPORT_DIR, file_name, as_attachment=True)
