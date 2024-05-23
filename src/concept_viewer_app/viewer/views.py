# Create your views here.
from django.core.serializers import serialize
from django.http import HttpResponse
from django.template import loader
from django.views.decorators.http import require_http_methods

from .models import Result
from .process import process


@require_http_methods(["GET"])
def get_result(request):
    # https://docs.djangoproject.com/en/2.0/topics/serialization/
    results = Result.objects.all()
    response = serialize('json', results)
    return HttpResponse(response)


current_table = {'min_df': 0.00, 'max_df': 1., 'min_tf': 0.00, 'max_tf': 1., 'frex': 0.5}


@require_http_methods(["GET"])
def index(request):
    global current_table
    current_table = {'min_df': 0.00, 'max_df': 1., 'min_tf': 0.00, 'max_tf': 1., 'frex': 0.5}
    results = Result.objects.all()
    template = loader.get_template('viewer/index.html')
    context = {
        'results': results,
    }
    return HttpResponse(template.render(context, request))


@require_http_methods(["GET"])
def update_table(request):
    global current_table
    current_table['min_df'] = float(request.GET.get('min_df', current_table['min_df']))
    current_table['max_df'] = float(request.GET.get('max_df', current_table['max_df']))
    current_table['min_tf'] = float(request.GET.get('min_tf', current_table['min_tf']))
    current_table['max_tf'] = float(request.GET.get('max_tf', current_table['max_tf']))
    current_table['frex'] = float(request.GET.get('frex', current_table['frex']))
    results = Result.objects.all()
    handler = process(**current_table, dataset=results[0].dataset)
    results = map(handler, results)
    template = loader.get_template('viewer/table.html')
    context = {
        'results': results,
    }
    return HttpResponse(template.render(context, request))
