CDF       
      obs    B   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?pbM���   max       ?�KƧ       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�K_   max       PͿ�       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��;d   max       <u       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?W
=p��   max       @F]p��
>     
P   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�z�G�    max       @v�\(�     
P  +   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @P            �  5d   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @��           5�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �bN   max       %          6�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B0j       7�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B0'u       9    	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @)�g   max       C���       :   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @)�0   max       C��<       ;   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          [       <   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          I       =    num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          ?       >(   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�K_   max       P���       ?0   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���?   max       ?�p��
=q       @8   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��;d   max       <49X       A@   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?aG�z�   max       @FJ=p��
     
P  BH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?��\(�   max       @v�33334     
P  L�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @L@           �  V�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�f�           Wl   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�       Xt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��_o�    max       ?�m\����     �  Y|   +   [               8   
         3         &       B      T            :         #   	                     %   
         '   	                           -            ;               F            !                  O��_PͿ�N)�NT��M�K_OQ��P�TxN�	gOAJ�O��QP7�O\�N��PS�9Pe[P^4�NAEO��O���N�N<�(P$��OE�wN�'<O��6N�ͲO��O��O���N�7M���OWLO̰�N�-N�+�O -�O��NI4OnL�OU�N0��O��O�^�M�+�O��ND��O�0kOb��Or�Nh�kP+,�N���N�49N� =O!�P '/O��O��Oy�O,ѦN�&�NA�O�zbNK��O$JUO'=<u<49X;ě�;�o;o��o�o�o�t��t��49X�49X�T���e`B�u��o��C���t���t����㼬1��j��j��j��j�ě����ͼ��ͼ��ͼ��ͼ��ͼ��o�\)�\)��P��P��P��w��w�#�
�#�
�'49X�<j�H�9�H�9�L�ͽY��Y��Y��u�y�#��7L��O߽�hs���㽛�㽛�㽧��1��1��vɽ�;d��;d
#+?JKF></#���#F{������bF<#���46ABBOSWOB<644444444#*0<EE<60#"	


										FHTafmuzzxtlaTHFA>AF������������������}���������������������=ESOPKN[]ghgiee[NJE=����
#)--#
������)6B[tp}�����[G0+}�����������������y}��������������������6B[h���������hO6,*06��������
��������et������������th][e��

��������������������� ��������������������������./1<HSU]WUH=<5/+**..�������������������������������������)5BNgktwvtgc[R5-)"knpz��������zzrnkkkk
*6Catyrh\O6*~�������������}}~~~~�)5BHNQOB5)���+0BN[goqm_[[NB7))%%+�������������������������� �����������������������������#0<EPUUPKII<;50#cimt�������������tgc���	������������������������������wz�������������zwtww 
)5BGNQSNJ@)�� FHNUalmaWUSHFFFFFFFF]adjlz����������zna]+7BO[_hklkjh^[OEC?6+;<CHLRSPHF<<;;;;;;;;QUXanz���������zaUQQGVaqz�������zsm`RNJG���������������������������	
�����������������������������#<Uab\UH</#��������������������������������������������������������������������������������RUV\bnooonkbUTRRRRRR��������������������!#06<AIJIHA<0#" !!!!stv��������������wts��������������������|����������������}||�����������������������������������������������������������������
�������mnvz�����znbmmmmmmmm����$)$�����O[gtxutg[UOOOOOOOOOOLNQV[\gt������tpg[NL����������������������������)�6�B�[�t�v�u�e�[�O�B�)����������Z�A�!��A�g��������������	���a�\�U�U�U�]�a�n�w�r�n�d�a�a�a�a�a�a�a�a�/�*�"���"�/�7�;�F�<�;�/�/�/�/�/�/�/�/ŹŵŹ��������������ŹŹŹŹŹŹŹŹŹŹ��ƽƳƲƳƸ������������������������������ƧƎ�^�b�uƁƧ�������$�5�.�6�3�-������������������&�*�6�C�9�6�*�����M�A�(����������4�A�M�Z�f�r�k�f�Z�MìÕÓÈÌÐÙàìù����������������ùì�����z�_�O�������л���'�*���ܻû����������z�t�s�n�x�����������»Ļ���������à×ÓÏÎËÊÓàäì÷ùýùùðìàà�ƾ��ľоʾ������Ǿ���8�S�X�Y�G�;�	�ƿ`�L�@�A�G�T�b�m�t�����ĿѿտϿҿ˿��y�`�Y�E�6�:�:�+�)�M�f�������������������Y�����������
��
�
����������������������h�\�O�C�*�������*�6�C�O�\�e�k�n�h�&�� �(�4�M�Z�f�s������������s�Z�M�4�&�
�	�
�����#�/�6�6�<�=�<�;�/�#��
�
�����������������������������������������	������	�"�/�a�m�z�y�m�d�\�H�;�"��	������������	�������	����������������������������������������������.�"�������	�"�.�;�G�S�T�P�L�N�G�;�.�Z�P�N�I�N�N�Z�Z�g�g�l�s�t�w�s�g�Z�Z�Z�Z���������g�X�N�O�s���������������������Կ�ݿʿ������Ŀ����5�A�K�L�A�>�(����	����׾˾žʾ׾���	��"�3�;�G�>�.��	�;�5�/�-�-�/�;�?�H�T�U�Y�V�T�N�H�;�;�;�;�H�G�<�2�/�.�/�<�>�H�K�I�H�H�H�H�H�H�H�H���������������������Ľн߽�ݽнʽʽ�������ŹšŎŇ�{ŇŐŠŭ�����������������׾Ҿʾžʾ׾�����������׾׾׾׾׾׽нν̽нԽݽ������������ݽннннн�àÛØÚàìíù��������������ýùìàà�<�0�#�����!�#�0�3�<�I�^�f�l�m�b�I�<�	����������	������	�	�	�	�	�	�	�	�T�J�H�;�/���������	��"�/�;�F�K�R�`�T����������������������������������������FFFFFFF$F1F5F1F%F$FFFFFFFFF=F6F=FCFFF@F1F"FFF1FJFaFtF{F~F{FpFcF=������������(�5�>�G�N�V�Z�A�5�(��B�@�6�)�%�)�6�?�B�F�F�E�B�B�B�B�B�B�B�B���|�}���������Ŀѿ�����߿ʿ�������������������������������������������������EuEjEpE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eu�g�N�@�C�A�1�5�A�N�Z�g�r�������������y�g�I�?�@�I�M�U�f�n�z�{�Ńŀ�{�z�q�n�b�U�IŔōōŎŔŚŠŭűųŭŠŔŔŔŔŔŔŔŔ���e�[�S�W�e�����ֺ���"������ɺ����û��������������ûɻлۻԻлûûûûû��6�5�)�'�)�1�6�B�O�Z�[�c�[�Y�O�B�6�6�6�6�_�Z�X�S�N�S�]�_�h�l�x�{����x�l�_�_�_�_¿¹²¦¦«²¿����������������¿�6�-��������)�6�B�[�h�k�r�t�o�f�[�B�6ĚďčĆāččĚĦĳĸĽĿ��ĿĳİĦĚĚ���
��
�
����#�.�0�<�>�?�=�<�0�#�ĿĶĦġĦĳ������������������������Ŀ�ּмʼ����������ʼڼ�����������ּ��������������!�%�����������������!�&�*�!���������������l�c�_�l�y�������Ľн߽ڽнʽĽ���������������������������a�U�H�<�0�/�&�#�'�/�3�<�H�M�R�U�\�d�f�a�4�(�)�4�9�@�M�N�Y�f�l�q�n�f�c�Y�M�@�4�4 B [ O T a  H ] �   i ( F O 6 W a & T 4 ( F a ? D @ | Y V D n 5 D d 4 < R U c 6 < l 9 ^ g F E \ ] G _ O 0 2 G +  ) A 8 q f G Y U D  ]  \  V  �    �  �    +  7  �  �  4  �  �  @  �  �  �  
  S  �  �  �    �    �  l  �  =  �    �  �  &  �  y  a  �  L    A  O  �  Z  �    l  m  �  �  �  �  �  7  5  B  �  �  l  G  t  �  �  I��/�������
%   �D����`B��o��C��u�\)�}�t����T���@���ě�����+��/��㽣�
�t���`B�ixսo�Y��T���e`B��`B��h�H�9��C��8Q�#�
�q������<j�e`B�m�h�<j��O߽�7L�u����aG�����}󶽍O߽ixս�S���7L���P��\)��bN��E���9X�\���\����
=����1'�I�B�B&�HBS�B%�nB�A��B-�B��B� B�yB�uB �B��BN�B+�>B �oB��B�)B �eB�"B�B�xB+6B��B0jB\�B�BRHBNB�B!9�B&	tB
��B({B"O�B "B��B�B��B�lBB�B�A�̄B�BOB"B.�B�ZB��B-�BeB'��B̡B%�B��BXYB_B9/BعB*rGB-�B2�B��B	%B	w�BRB��B&�vB@�B%�.B�YA���B��B�JB�"B�oBBvB ?�B�#B?�B*�B ��B��B��B �B>aB�B��B�B��B0'uBk�BIUBM�B7B�^B!E�B&6�B4�B@B":B �'B��B�4B�B?�B@~B�#A��B��B�B�|B?�B��B@�B�BCnB'�B�UB%��B3�B3�B
�UB>NB �B*@�B.:B@B�FB	? B	�|B@�A�SA�tvA��_A�0A���B�6B�NA�:�A=eA�p@�Vn@�%Aˇ�A[��AqeJ@�YA��:B |A?�A�l�A��_A��8AX�PA���A_��A��ZA��4A�8.AZ@NA�.�AË�A# �A�
AU��A+�qA�HuA�+AZO=A���A���C��wC���A�a�A��gAw#�A���C�$�A���A��A�j�@)�g@�tAؽr@� oA��A�J�A�'�A�c�A�	�AQ�A�l@fb�A!�A��=A�_�@�A�AצA�� Aƀ�A���A���B�B:�A���A=A�~�@��	@���A�\�A\�An�=@�T#A�v�B �A>��A�݇A��!A�i�AZ�rA�ߊA^��A��gA���A~��AY��A�vA��VA!/A�]�AS�A,�A�iCA�N�AZ��A��3A�5C��C��<A�g9Aׁ AufMA�i�C�'�A��BA�~ZA��T@)�0@��kA؂�@�m�A��A؄�A���A괫A�i�AnA	
@d �A#�A��A��@�9�   +   [               9   
         4         '   !   C      T      	      :         #   	                      %            (   
                           .            ;               G            "            	            I               ;            ;         5   3   1         #         )         %      +   %               %                           '         !      %   #         3               #                     !               ?                           ;         !   3   #         #                           %                                          %         !                  1                                    !         Ou��P���N)�N1�M�K_O��O���N�	gN��)O[,P7�OM�9N�kEO��PPe[O���NAEOf�O���N�N<�(O��zOE�wN�'<O�}^NM4�OGO��O��fN�7M���O?|�OBH�N�-N�+�O -�OQRNI4O>�OD�N0��O��fO��M�+�O��ND��N�&�O9�Or�Nh�kPi�N���N�49N� =O!�O�	�O��O��OQ�O�1N�&�NA�O�zbNK��O$JUO'=  �  u  �  ;  *  n  �  �  �  7  |  )  x  
    l  �    @  I    �  �  �  �    �      �  �  �  �  �  �  8    �    �  �  �  �  �  �  �  �  �  �  K  �  @  �  (  &  
�  #  �  @  t  j  N  �  (  b  �<49X�D��;ě�;D��;o�#�
���o�49X�u�49X�D���u���u�t���C��,1��t����㼬1�0 ż�j��j�������t����ͼ�`B���ͼ��ͽo�49X�\)�\)��P�8Q��P�'#�
�#�
�,1�,1�49X�<j�H�9��t��P�`�Y��Y��e`B�u�y�#��7L��O߽��-���㽛�㽟�w��-���1��1��vɽ�;d��;d	
#(;GIFD</(#

	��0Un�������_0
���46ABBOSWOB<644444444#+0<BA<00-#	


										EHKTalmsqmkbaTNHEBEE����������������������������������������@BNNX[]bda`[ONIB@@@@�����
!#))#
�����)6B[tp}�����[G0+��������������������������������������67<BO[h�����~{qh[B66��������
��������q���������������unnq��

���������������������������������������������������./1<HSU]WUH=<5/+**..��������������������������������������)5BNgktwvtgc[R5-)"knpz��������zzrnkkkk*6COXagjc\C6-	�������������������()05?BIKHB@5)+0BN[goqm_[[NB7))%%+�������������������������� �����������������������������#0<CIOUROJIF<70#z��������������}xvuz���	������������������������������wz�������������zwtww)5:BFGB:)FHNUalmaWUSHFFFFFFFFmrw{�����������{nilm/69BOZ\hjlkih\OFD@6/;<CHLRSPHF<<;;;;;;;;Unz����������znaVRRUGXasz�������zrma[TJG���������������������������	
����������������������������� #/3<CE=<1/+#     ��������������������������������������������������������������������������������RUV\bnooonkbUTRRRRRR��������������������!#06<AIJIHA<0#" !!!!stv��������������wts��������������������|����������������}||�����������������������������������������������������������������
�������mnvz�����znbmmmmmmmm����$)$�����O[gtxutg[UOOOOOOOOOOLNQV[\gt������tpg[NL��������������������������)�6�B�O�[�s�q�h�b�[�O�B�6�)��������q�X�A�5�2�?�g������������������a�\�U�U�U�]�a�n�w�r�n�d�a�a�a�a�a�a�a�a�/�,�"�� �"�/�3�;�=�;�8�/�/�/�/�/�/�/�/ŹŵŹ��������������ŹŹŹŹŹŹŹŹŹŹ������ƸƷ��������������������������������������ƺƵƶ��������������!��������������������&�*�6�C�9�6�*�����M�J�A�@�4�+�4�A�M�Z�f�m�f�f�Z�P�M�M�M�M��ùìÞ×××àáìù�����������������Ż����z�_�O�������л���'�*���ܻû����������{�x�u�t�p�x�������������»�������àÚÓÐÏÌÓÓàâìôùüù÷îìàà�.�"���߾�������	��"�.�8�?�C�;�.�`�L�@�A�G�T�b�m�t�����ĿѿտϿҿ˿��y�`�Y�M�F�D�G�E�M�Y�f���������������r�f�Y�����������
��
�
����������������������6�*������*�6�C�O�S�\�^�c�d�\�O�C�6�&�� �(�4�M�Z�f�s������������s�Z�M�4�&�
�	�
�����#�/�6�6�<�=�<�;�/�#��
�
�����������������������������������������"������#�/�;�T�a�f�l�e�\�T�H�;�/�"������������	�������	����������������������������������������������"��	��������	��"�.�;�C�K�K�G�;�:�.�"�Z�X�O�Q�Z�g�j�q�o�g�Z�Z�Z�Z�Z�Z�Z�Z�Z�Z������������������������������������������ݿʿ������Ŀ����5�A�K�L�A�>�(�����׾ξɾ׾���	��"�/�;�@�8�.�"��	����;�5�/�-�-�/�;�?�H�T�U�Y�V�T�N�H�;�;�;�;�H�G�<�2�/�.�/�<�>�H�K�I�H�H�H�H�H�H�H�H�������������������������Ľ̽нݽнǽǽ�ŭťŠŝŠŢŭŹ��������������������Źŭ�׾Ҿʾžʾ׾�����������׾׾׾׾׾׽нν̽нԽݽ������������ݽннннн�àÛØÚàìíù��������������ýùìàà�I�<�0�#�!�����#�0�<�I�X�^�b�b�U�T�I�	����������	������	�	�	�	�	�	�	�	�/�"��	����������	��"�/�;�D�H�M�H�;�/����������������������������������������FFFFFFF$F1F5F1F%F$FFFFFFFFFBFHFDF1F#FF!F1FJFVF`FqFyF}FyFyFoFcFVFB����� ������(�5�=�F�N�T�X�N�D�(��B�@�6�)�%�)�6�?�B�F�F�E�B�B�B�B�B�B�B�B���|�}���������Ŀѿ�����߿ʿ�������������������������������������������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��g�`�Z�N�C�E�U�Z�g�s�����������������v�g�I�?�@�I�M�U�f�n�z�{�Ńŀ�{�z�q�n�b�U�IŔōōŎŔŚŠŭűųŭŠŔŔŔŔŔŔŔŔ���^�V�\�e�����ֺ���!�������ɺ����û��������������ûɻлۻԻлûûûûû��6�5�)�'�)�1�6�B�O�Z�[�c�[�Y�O�B�6�6�6�6�_�Z�X�S�N�S�]�_�h�l�x�{����x�l�_�_�_�_¿¹²¦¦«²¿����������������¿�1�#������6�B�[�e�h�o�q�m�b�[�O�B�1ĚďčĆāččĚĦĳĸĽĿ��ĿĳİĦĚĚ���
��
�
����#�.�0�<�>�?�=�<�0�#�ĦĤĪĳĶ�����������������������ĿĳĦ�ʼ����������Ƽʼּ��������ּʼʼ��������������!�%�����������������!�&�*�!���������������l�c�_�l�y�������Ľн߽ڽнʽĽ���������������������������a�U�H�<�0�/�&�#�'�/�3�<�H�M�R�U�\�d�f�a�4�(�)�4�9�@�M�N�Y�f�l�q�n�f�c�Y�M�@�4�4 = V O Z a  T ] N $ i $ C ; 6 Q a ( T 4 ( I a ? < E ; Y S D n )  d 4 < I U G 4 < j 8 ^ g F - S ] G b O 0 2 G &  ) B , q f G Y U D  �  }  V  P    (  �    �  �  �  �  
  �  �  q  �  �  �  
  S  �  �  �  X  o  *  �  .  �  =  �  �  �  �  &  �  y  �  �  L  
  6  O  �  Z  �  �  l  m  }  �  �  �  �  �  5  B  �  ,  l  G  t  �  �  I  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  k  �  �  �  �  n  Q  /    �  �  �  g  .  �  �  [    ;  �  �  E  o  u  p  \  ?  !  �  �  ~  0  �  �  $  �  �  $  H    �  �  �  �  �  }  l  [  B  )    �  �  S    �  �  v  :   �  3  5  7  8  :  <  ?  B  D  G  P  `  o  ~  �  �  �  /  h  �  *  %  !      �  �  �  �  �  x  X  2    �  �  {  A    �  2  F  W  c  k  n  l  e  W  A  "  �  �  �  L    �  m  =    M  2      N  q  �  �  �  �  �  �  �  v  7  �  ]  �  L  *  �  �  �  �  �  ~  w  j  [  E  )    �  �  �  M    �  �  a  �  �  �  �  �  �  �  �  �  �  �  |  k  Z  F  2     �   �   �  �    $  1  6  4  %    �  �  �  T    �  }  B    �  $  �  |  d  X  J  +    �  �  �  �  �  �  �  �  Z    �  _    �    '      �  �  �  �  t  K  J  =  '    �  �  s  7  �  �  9  ]  v  w  o  b  O  1  
  �  �  d  &  �  �  ~  E    �  �  �  �  �  �  �  �  �    
  �  �  �    H  �  �  5  �  _         �  �  �  �  �  �  }  W  -  �  �  �  L  	  �  �       �  �  .  H  V  c  k  _  ]  P    �  +  �  Q    �  �  C  �  �  �  �  �  �  �  �  �    m  Z  E     �  �  �  x  (  �  m    �  �  �        �  �  �  +  �  5  
�  	�  	    �  �  %  @  @  =  9  1  #    �  �  �  �  �  ]  ?  .    �  �  �  �  I  0       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �      !    (    �  �  �  U    �  �  ,  �  g  �  �  #  �  9  g  �  �  �  �  �  �  �  �  �  �  e  3  �  �  C  �  �  .  �  �  �  �  �  �  {  g  S  A  /      �  �  �  �  `  5      �  �  �  �  �  �  �  �  ~  Z  7        �  �  �        �  �  �  �  �  �  �  �  �  G  �  �  x  k  [  /  �  �  m    	                  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Y  K  �  �  �  �  �  �  �  �  �  N    �  �  �  $       �  �  �  �  �  �  _  4    �  �  �  j  @    �  �  �    
      �  �  �  �  �  t  H    �  �    �  C  �  =  �  �  �  �  �  �  �  �  �  }  n  _  Q  B  1       �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  j  \  O  A  �  �  �  �  �  m  Y  @  )        �  �  �  �  �  b  =  �  1  m  �  �  �  �  �  �  �  �  �  �  ^  0  �  �  5  �  �  �  �  �  �  �  �  �  �  k  M  .    �  �  �  �  d  ;    �  s  �  �  �  �  �  �  �  �  �  �  �  �  �  w  ]  C  '     �   �  8  1  (      �  �  �  l  ;    �  �  e  1    �  �  -  �  �  �  �  �    �  �  �  �  |  M    �  �  `  �  S  �  �  �  �  �  �  �  �  u  `  F  (    �  �  �  V    �  �  m  0   �    �          �  �  �  �  �  n  J  %  �  �  a    �  �  �  �  �  �  �  �  �  �  s  R  1    �  �  �  �  �  `  -  �  �  �  �  �  �  �  n  S  6    �  �  �  �  �  m  S  E  7  *  �  �  �  �  �  N  	  �  y  %  �  &  M  >     �  �  9  �  �  o  ~  v  h  S  2    �  �  I    �  j    �  �  &  �    X  �  �  �  �  �  �  �  �  �  �  U  '  �  �  �  \  %  �  �  v  �  �  |  {  g  J  /  '      �  �  �  F    �  x    �  .  �  �  �  �    u  k  ^  O  A  3  &           �        �  �  �  �    E  x  �  �  �  �  �  �  e    �    E  d  p  �  �  �  �  �  |  h  P  7  .  $      �  �  �  �  �  �  �  �  �  �  t  d  S  I  &  �  �  �  r  <     �  �  V  �  �  /  K  B  :  2  *  "          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  d  3  �  �  �  G  �  }  �  �  �  3  ]  p  @  =  ;  5  )      �  �  �  �  �  �  k  L  (     �   �   ~  �  �  �  z  ^  @    �  �  �  q  @    �  �  <  �  X  �  b  (  !             �  �  �  �  �  �  �  �  �  y  c  N  9  &  
  �  �  �  �  =  �  �  A  �  �  �  l  G  #  �  �  �  v  
�  
�  
�  
�  
�  
�  
�  
_  
  	�  	@  �  t    �    B    �  �  #  !            �  �  �  �  �  �  `  6    �  �  �  k  �  �  �  �  �  �  �  �  b  D  $    �  �  �  ^  .  �  z   �  0  ;  @  :  4  +  "      �  �  �  �    Z  2  	  �  �  �  W  a  o  s  g  U  =    �  �  �  \  &  �  �    P  �  �   �  j  U  N  O  3    �  �  �  z  N    �  �  �  �  [    �  �  N  "  �  �  j  *  �  ~  F    �  \  �  �  *  �  S  �  q  �  �  �  �  �  �  �  �  �  s  U  4  '    �  �  �  ~  �  �  �  (        �  �  �  �  �  �  �  �  o  W  @  '    �  �  �  b  O  <    �  �  x  E    �  �  T  �  �    �  (  �  �    �  �  �  �  �  �  t  W  2    �  {  #  �  g  	    �  �  