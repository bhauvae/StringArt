#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <cmath>
#include <stdexcept>

using namespace cv;
using namespace std;

Mat crop_image_to_circle(const string &image_path, const string &output_path, int max_length = 3840)
{
    Mat image = imread(image_path, IMREAD_UNCHANGED);
    if (image.empty())
    {
        throw runtime_error("Could not open or find the image");
    }

    int height = image.rows;
    int width = image.cols;

    if (max(width, height) > max_length)
    {
        if (width > height)
        {
            int new_width = max_length;
            int new_height = static_cast<int>(height * max_length / width);
            resize(image, image, Size(new_width, new_height), 0, 0, INTER_AREA);
        }
        else
        {
            int new_height = max_length;
            int new_width = static_cast<int>(width * max_length / height);
            resize(image, image, Size(new_width, new_height), 0, 0, INTER_AREA);
        }
    }

    height = image.rows;
    width = image.cols;
    Point center(width / 2, height / 2);
    int radius = min(center.x, center.y);
    Mat mask = Mat::zeros(height, width, CV_8UC1);
    circle(mask, center, radius, Scalar(255), -1);

    Mat result;
    if (image.channels() == 1)
    {
        image.copyTo(result, mask);
    }
    else
    {
        Mat masked_image;
        vector<Mat> channels;
        split(image, channels);
        for (auto &channel : channels)
        {
            Mat masked_channel;
            channel.copyTo(masked_channel, mask);
            channels.push_back(masked_channel);
        }
        merge(channels, result);
    }

    int square_size = radius * 2;
    int crop_x = (width - square_size) / 2;
    int crop_y = (height - square_size) / 2;
    Mat cropped_square = result(Rect(crop_x, crop_y, square_size, square_size));

    Mat output_image = Mat::zeros(square_size, square_size, image.type());
    cropped_square.copyTo(output_image(Rect(0, 0, cropped_square.cols, cropped_square.rows)));

    imwrite(output_path, output_image);

    return output_image;
}

Mat radon_string(double s_string, double alpha_string, int radius)
{
    int size = radius * 2;
    Mat radon_transform = Mat::zeros(size, size, CV_64F);
    vector<double> proj_angle(size);
    vector<double> proj_pos(size);

    for (int i = 0; i < size; ++i)
    {
        proj_angle[i] = i * 180.0 / size;
        proj_pos[i] = i - radius;
    }

    Point max_index(0, 0);
    for (int alpha_idx = 0; alpha_idx < size; ++alpha_idx)
    {
        double alpha = proj_angle[alpha_idx];
        double sin_angle = sin((alpha - alpha_string) * CV_PI / 180.0);
        if (abs(sin_angle) == 0)
        {
            max_index.y = alpha_idx;
            continue;
        }

        for (int s_idx = 0; s_idx < size; ++s_idx)
        {
            double s = proj_pos[s_idx];
            if (s == s_string)
            {
                max_index.x = s_idx;
            }

            double evaluator = (s * s + s_string * s_string - 2 * s * s_string * cos((alpha - alpha_string) * CV_PI / 180.0)) / (sin_angle * sin_angle);
            if (evaluator > radius * radius)
            {
                radon_transform.at<double>(s_idx, alpha_idx) = 0;
            }
            else
            {
                radon_transform.at<double>(s_idx, alpha_idx) = 1 / abs(sin_angle);
            }
        }
    }

    radon_transform.at<double>(max_index.x, max_index.y) = radon_transform.at<double>(max_index.x, max_index.y) * 1.1;
    normalize(radon_transform, radon_transform, 0, 1, NORM_MINMAX);

    return radon_transform;
}

void plot_sinogram(const Mat &sinogram)
{
    Mat display;
    normalize(sinogram, display, 0, 255, NORM_MINMAX);
    display.convertTo(display, CV_8U);
    applyColorMap(display, display, COLORMAP_HOT);
    imshow("Sinogram", display);
    waitKey(0);
}

pair<vector<pair<double, double> >, int> get_strings(const string &img_path, int num_of_anchors)
{
    Mat image = imread(img_path, IMREAD_GRAYSCALE);
    if (image.empty())
    {
        throw runtime_error("Could not open or find the image");
    }

    image = 255 - image;

    int height = image.rows;
    int width = image.cols;
    Point center(width / 2, height / 2);
    int radius = min(center.x, center.y);

    vector<double> theta(radius * 2);
    for (int i = 0; i < radius * 2; ++i)
    {
        theta[i] = i * 180.0 / (radius * 2);
    }

    Mat sinogram_img = Mat::zeros(radius * 2, radius * 2, CV_64F);
    for (int i = 0; i < radius * 2; ++i)
    {
        for (int j = 0; j < radius * 2; ++j)
        {
            double angle = theta[j] * CV_PI / 180.0;
            double sum = 0;
            for (int k = 0; k < radius * 2; ++k)
            {
                int x = center.x + (k - radius) * cos(angle);
                int y = center.y + (k - radius) * sin(angle);
                if (x >= 0 && x < width && y >= 0 && y < height)
                {
                    sum += image.at<uchar>(y, x);
                }
            }
            sinogram_img.at<double>(i, j) = sum;
        }
    }

    normalize(sinogram_img, sinogram_img, 0, 1, NORM_MINMAX);

    vector<pair<double, double> > strings;
    Mat len_adj_sinogram = Mat::zeros(sinogram_img.size(), CV_64F);
    for (int n = 0; n < num_of_anchors; ++n)
    {
        for (int s = 0; s < radius * 2; ++s)
        {
            if (s == 0)
            {
                len_adj_sinogram.row(s).setTo(0);
            }
            else
            {
                double len_of_proj = 2 * sqrt(radius * radius - (s - radius) * (s - radius));
                len_adj_sinogram.row(s) = sinogram_img.row(s) / len_of_proj;
            }
        }

        Point max_index;
        minMaxLoc(len_adj_sinogram, nullptr, nullptr, nullptr, &max_index);

        int s_idx = max_index.y;
        int theta_idx = max_index.x;

        double alpha = theta[theta_idx];
        double s = s_idx - radius;

        double theta1 = alpha - acos(s / radius) * 180.0 / CV_PI;
        double theta2 = alpha + acos(s / radius) * 180.0 / CV_PI;

        strings.emplace_back(theta1, theta2);

        Mat sinogram_string = radon_string(s, alpha, radius);
        sinogram_img -= sinogram_string;
        sinogram_img.setTo(0, sinogram_img < 0);

        cout << "String " << strings.size() << ": " << theta1 << " to " << theta2 << endl;
    }

    plot_sinogram(sinogram_img);

    return {strings, radius};
}

Mat create_string_art(const vector<pair<double, double> > &strings, int radius)
{
    Mat canvas = Mat::zeros(radius * 2, radius * 2, CV_8UC1);
    Point circle_center(radius, radius);

    for (const auto &string : strings)
    {
        double theta1 = string.first * CV_PI / 180.0;
        double theta2 = string.second * CV_PI / 180.0;

        Point pt1(circle_center.x + radius * cos(theta1), circle_center.y + radius * sin(theta1));
        Point pt2(circle_center.x + radius * cos(theta2), circle_center.y + radius * sin(theta2));

        line(canvas, pt1, pt2, Scalar(255), 1);
    }

    return canvas;
}

int main()
{
    try
    {
        Mat cropped_image = crop_image_to_circle("art.png", "test.png");
        auto [strings, radius] = get_strings("test.png", 900);
        Mat art = create_string_art(strings, radius);
        imwrite("test_900.png", art);
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}