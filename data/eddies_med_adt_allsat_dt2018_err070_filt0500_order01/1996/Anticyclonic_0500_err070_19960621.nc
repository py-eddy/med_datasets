CDF       
      obs    <   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?Ƨ-      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N�   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �H�9   max       =���      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�\(�   max       @E��G�{     	`   |   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�\(��    max       @vyp��
>     	`  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @&         max       @P            x  3<   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @��          �  3�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �<j   max       >Q�      �  4�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��r   max       B+�)      �  5�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B+�9      �  6�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��:   max       C���      �  7t   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C���      �  8d   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  9T   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ?      �  :D   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1      �  ;4   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N�   max       Pk"�      �  <$   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_خ   max       ?�\����>      �  =   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �H�9   max       =��#      �  >   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�(�\   max       @E��G�{     	`  >�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�Q�    max       @vyp��
>     	`  HT   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @L�           x  Q�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @ˣ        max       @�3`          �  R,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         BM   max         BM      �  S   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�R�<64   max       ?�\����>     �  T                  D      �            $         C   <      h            
   -   3                     !         �                         W   #                  	         3      
   
   (            N�N!rN3բN4��N�m+P��N��P���N�TO��N\VpO�:hOD1�N���Pr<VP �N���O�SN�X�N���N:�^N<��O��4O�VbN��cN�ǡN�j�N��{O���O�?�O���N���O���O�HTO�)N�iP06N_bfN�boN�m}O`�P�;P&-iOU�N�6N�dUO�5�O�>6N,q�N`��Ox�-O��{OC><N��|O��Oп�N��N��ND��N��H�9��`B�u�o��o:�o;��
;�`B;�`B<t�<49X<D��<e`B<e`B<�o<�o<�C�<�t�<�t�<�t�<�t�<�t�<�t�<��
<�1<�1<�1<�j<���<���<�/<�/<�`B<�h<��<��=o=o=+=C�=C�=\)=�w=#�
='�=,1=49X=49X=49X=@�=D��=P�`=T��=u=��=�7L=�7L=�7L=�-=�����������������������@ABFOYYSOB@@@@@@@@@@���������������������������������������������������������������)1B[t�v[5���mjlpt�����������tmm�����6BTSIB=3�����������������������#,0<IO]ab^UJ<0-&e_cgst����ztkgeeeeee)BNTfstpg[NB5,"	"/;@A<;502/"V[[dgt}����xtg`[VVVV����)DLJG5)��������%-BIMM5)!������������������fgt��������������qjf..//6<HLRMHF</......)*//+)~�������������~~~~~~����������������������������������������wuusv{������������zw���� ���!)*-+) 

##%055400#


��������������������trrvz�������������zt�{y|����������������
'(+-68;:74".+,66BJOUTOLB6......&8<HUalnrnaUH</#��������
" 
�������)57<>:5)
�����������������������{������������������{������������������������������������������������������������"'*/26<HPWZZWTH<.&#"^^lt�������������tm^)���������)10))bchlt|����������tjhb"')*)(�������������������������+5ABFB5)�GECNPTgt������tg[QNG��������������������<BDN[ghgf[NB<<<<<<<<"(35BINRTSNFB5�����
#/5762/+#!
��������������������������������	���������������������������������� " ����\WWalnz}zna\\\\\\\\z}~�����������zzzzzz.)$/6<ADA<7/........#'##�����
��
�����������������������������y�������������y�u�r�y�y�y�y�y�y�y�y�y�y�S�_�a�`�_�T�S�O�F�@�F�H�S�S�S�S�S�S�S�S��������������w������������������������'�3�=�>�<�6�3�'����������%�'�'�����5�N�W�T�A�-�$���Ŀ�������������3�@�L�Y�^�e�g�j�l�e�Y�V�L�@�;�3�/�1�3�3�Y�d������¼˼ϼʼ����r�@�4�'���4�M�Y�����������������������������������������������������������������o�f�`�b�f�r��m�y�����������������y�o�m�k�m�m�m�m�m�m���������������ñãâçõ�����������;�H�T�a�l�s�y�y�m�a�T�Q�H�;�/�&�#�/�4�;�b�n�n�{�}ŀ�{�v�n�b�^�U�S�S�U�Y�b�b�b�b�0�<�Uŀň�}�l�U�I�#�
���������������
�0āĚĦĳĿ����ĳĦčā�t�h�P�I�B�:�:�Hā�4�>�A�M�Q�M�K�A�9�4�(�&�����(�+�4�4�O�R�[�h�d�[�O�=�6�)����������6�B�O�����������������������������������޿y�������������������y�x�m�i�m�t�y�y�y�y��������� �
��
�������������������������#�/�<�>�<�:�1�/�*�#� ��#�#�#�#�#�#�#�#ù����������������úìàÓÇ�~ÀÌÓìù�tāčĦĳĿ��������ĿĳĦĚčĀ�t�l�h�t�;�G�R�T�`�b�`�T�G�;�9�.�"���"�'�.�7�;������	������׾ԾʾȾʾ׾���������������������׼��������E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E�E�E��5�A�Z�s�{�������������������s�N�?�0�(�5�	��"�.�G�T�Z�S�G�;�"��	��ھھ����	�����	�/�<�H�T�a�a�T�G�;�"��������������m�z���������������z�p�m�f�g�m�m�m�m�m�m��"�/�;�H�L�M�G�7�"��	��������������D{D�D�D�D�D�D�D�D�D�D�D�D�D{DkDdDhDhDsD{�s�t�u�|�z������s�f�Z�V�R�R�V�Z�a�f�q�s���������������������������������������������������������������������s�l�w�s�����Z�f�f�]�Z�N�E�A�6�A�N�R�Z�Z�Z�Z�Z�Z�Z�Zàìù��������ùìàÓÉÓßàààààà�{ŇŔŠŦŭűŭšŠŔŇ�{�y�v�y�{�{�{�{�(�4�A�S�Z�f�r�j�Z�M�A�4�(������"�(�'�4�Y�g�p�s�n�d�M������ܻ˻лۻ���'�8�2�6�'���'�3�@�e�~�������������~�e�8�����������ĿǿɿĿ����������������������h�uƁƄƎƐƎƁ�u�p�h�]�\�U�\�d�h�h�h�h�a�g�n�v�t�n�e�a�_�U�T�P�U�]�a�a�a�a�a�a�	��"�/�;�D�L�I�;�*�"���	�������� �	�����	���"�-�.�%��	������������������ּ�������ؼּмּּּּּּּּּֽ����ĽʽͽĽĽ����������������������������������������#���������������ƿ�̿`�m�y�������������y�m�`�G�1�.�%�.�;�N�`�(�4�A�F�M�R�V�R�M�A�4�(������� �(�5�A�G�N�Z�Z�_�Z�N�D�A�5�(�!�'�&�(�.�5�5�F�_�p�x�������������������x�_�S�F�:�9�F����������������������º²¬®¹¿�˻����ûλû»����������������������������лܻ����� �����߻ܻл˻ûллллл�FFF$F'F$F"FFE�E�E�FFFFFFFFF�6�C�O�O�O�M�C�6�6�-�6�6�6�6�6�6�6�6�6�6 M F T 7 m = F : T  N H A ) & 5 " A B / ] H F ; r U ' Y � > [   $ & ` U 2 8 V + B 3 F ' R M 0 & V \ K J * 9 q - B 9 \ 8  %  =  t  F  6      A  6  �  �  �  �  �  �  l  �  c  �  �  �  ]  :  �  �  �  �  �  1  d  �  �  r  @  �  ?    g  �  �  �  �  �  I  �  �      V  �    b  �    �  �  �  �  G  �<j�ě��D����o<t�=�C�<��
><j<#�
<��<�t�=@�<���<���=���=���=o=���=t�<���<��
<�/=�%=�hs<ě�<�`B<�h<�`B=8Q�=�P=u=,1=H�9>Q�=,1=�P=�o=�w=H�9=<j=�%=��#=��=q��=<j=<j=u=�%=T��=L��=y�#=��=��w=�O�=���=��=���=� �=��>$�B�B�TB"}�B)�RB!]BQB��B4B"�B&R]B	�BA��rB	vB��B%,Bx�B��B�rBéB=�B �B!�XB ±B��BɖB%'�B�&B P2BУB�,B8tB^B�BK�B!-xB��BG�B9�B�UB�BF�Bc�B�BE?BQ�B=
B	yBB�JB`�BI;BPKB�GB�B+�)B�SB��B��B�LBkjB�$B�]B"yMB)��B!�B�B��B@�B"o�B&?�B	�B�A��B	��B�B�B��B��B<�B�jB��B�gB"?�B �rB�\B�0B%?gB��B ?�B��B��B?�B8�B=�Br~B!F�B�7BC\BG]B��B��B��B�kB�cB?�B�B��B	�aB XB�B�BT�B�oBy�B+�9B�BDnB;�B��BC�A�;A�B@��*@�r�?��:A�?�t�@���@� &@�I�Am�A�
A��A� eA�~jA�q�A8�IA�AA���Ao8sA��[A�?A�nA��AcAU��AX8C��EA���A]�	A��`A��A��C�� AA��@	�A��xA�ĐA��A�7A:\�@���?���At|B��AƬ�A�V�A�AA8A%�B�$Aj��A8�A�$@��A���@���@���C���B �:A�jA�b@��7@��?��A�.?��m@�8<@�@@�BJAn2UAϪ�A��ZA�'lA��Aݡ�A8�aA�{�A��An� A�GAyA�}A�~
Ad�}ASoAC��A���A]fA� SA���A��C��"AAE�@��A��A��A�EA��A948@���?���AtB�B�AƓ�A���A��>A61A%��Bn�Ak
ZA:H�A��*@��A��a@�@��,C���B �X                  D      �            $         D   <      h      	      
   .   4         	            "         �         !               X   #                  	         3      
      (                              ?      ;            #         3   %      #                                 !   #   #         !         -               -   /                                       !                              1      %                     !                                             #                     )                  /                                                   N�N!rN3բN4��N^t�Pk"�N�?(O��~N�TO=��N\VpO���N��TN���O���O�jN�w�OT�N��EN�8DN:�^N<��O�O� �N��cN�ǡN�j�N��{N�HoO�?�O}ЌN#o}O���OS�O�N�iP'�N_bfN��N��O+�O�H�P&-iOU�N�6N�dUO�5�Ot�N,q�N`��Ox�-O(7OC><N��|O��O���N��N��ND��N�      �  �  M  �  <  �    Y    T  2  �  �  R  �    �  [    �  4  	#  �  �  �  �  t  &  �  �  �  2  �  e  �  �    $  R  	O  �  �  �  �  �  D  �  �  �  	Z  2  �  4  �      �  ��H�9��`B�u�o;o<�o;�`B=��P;�`B<T��<49X<�C�<�t�<e`B=0 �=�P<���=�t�<���<���<�t�<�t�<��<���<�1<�1<�1<�j=o<���<��=+<�`B=��#=o<��=+=o=\)=\)=�w=�C�=�w=#�
='�=,1=49X=8Q�=49X=@�=D��=��=T��=u=��=��=�7L=�7L=�-=�����������������������@ABFOYYSOB@@@@@@@@@@���������������������������������������������������������������)Bdkl`C5���mnqt���������tmmmmmm����)6<<61)������������������������"!#0:<IW\\YRI<80+#"e_cgst����ztkgeeeeee%#%)5BOanpmg[NB5/*)%"/;<=;3/)"V[[dgt}����xtg`[VVVV����)5<<:5)��������)4>95)����������������������������������������/./07<HKRLHE<///////)-.))~�������������~~~~~~����������������������������������������wwux}������������~zw���� ���!)*-+) 

##%055400#


��������������������yvwxz{�����������zyy�{y|����������������	)5698651'	346BOPOOB63333333333&8<HUalnrnaUH</#�������	

��������&)25;=85)����������������������������������������������������������������������������������������������������.(),/5:<DHSUWWTOH</.kilpt������������~tk)���������)10))bchlt|����������tjhb"')*)(�������������������������+5ABFB5)�KNNQX[gt������tg[TNK��������������������<BDN[ghgf[NB<<<<<<<<"(35BINRTSNFB5�� 
#&/132/'#
���������������������������������	��������������������������������������\WWalnz}zna\\\\\\\\z}~�����������zzzzzz.)$/6<ADA<7/........#'##�����
��
�����������������������������y�������������y�u�r�y�y�y�y�y�y�y�y�y�y�S�_�a�`�_�T�S�O�F�@�F�H�S�S�S�S�S�S�S�S��������������w������������������������'�3�:�9�3�2�'������'�'�'�'�'�'�'�'�Ŀ����-�9�;�4� ����ѿ������������ĺ@�L�Y�]�e�e�g�e�Y�L�@�@�3�4�@�@�@�@�@�@�f�r���������������Y�M�?�4�0�0�7�M�Y�f���������������������������������������������������������������r�j�e�f�j�r�|��m�y�����������������y�o�m�k�m�m�m�m�m�m��������������������øéçìú���������H�T�a�g�m�o�s�o�m�a�]�T�N�H�@�B�H�H�H�H�b�n�n�{�}ŀ�{�v�n�b�^�U�S�S�U�Y�b�b�b�b�<�I�S�b�k�g�\�S�D�0�#����� ����#�<�h�tāčĚĦĬħĦĜč�t�h�d�[�V�V�[�f�h��(�4�A�M�N�M�I�A�4�(����������6�B�D�O�O�O�F�B�7�6�)�������)�6�6�����������������������������������޿��������������y�y�m�y�z������������������������� �
��
�������������������������#�/�<�>�<�:�1�/�*�#� ��#�#�#�#�#�#�#�#àåùÿ������ùìàÓÎÇÆÇËÓÕÞàāčĦĳĿ��������ĿĳĦĚĐă�t�q�t�xā�;�G�R�T�`�b�`�T�G�;�9�.�"���"�'�.�7�;������	������׾ԾʾȾʾ׾���������������������׼��������E�E�E�FFFFE�E�E�E�E�E�E�E�E�E�E�E�E��A�N�Z�g�p�s���������s�g�Z�N�J�A�<�>�A�A�	��"�.�G�T�Z�S�G�;�"��	��ھھ����	�	��"�/�8�D�F�;�9�"��	���������������	�z�����������z�y�m�q�z�z�z�z�z�z�z�z�z�z��"�/�;�H�L�M�G�7�"��	��������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��f�s�t�z�x�����~�s�f�Z�W�S�R�W�Z�c�f�f���������������������������������������������������������������������t�m�z�v�����Z�f�f�]�Z�N�E�A�6�A�N�R�Z�Z�Z�Z�Z�Z�Z�Zìù��������ùìàÓàãìììììììì�{ŇŔŠŤŬŠşŔŇ�{�z�w�z�{�{�{�{�{�{��(�4�A�K�Z�f�k�f�d�M�I�A�4�(������'�4�@�M�T�Z�W�Q�M�@�4�'����������'�8�2�6�'���'�3�@�e�~�������������~�e�8�����������ĿǿɿĿ����������������������h�uƁƄƎƐƎƁ�u�p�h�]�\�U�\�d�h�h�h�h�a�g�n�v�t�n�e�a�_�U�T�P�U�]�a�a�a�a�a�a�	��"�/�;�D�L�I�;�*�"���	�������� �	��������"�$�,�-�$��	������������������ּ�������ؼּмּּּּּּּּּֽ����ĽʽͽĽĽ����������������������������������������#���������������ƿ�̿y������������}�y�m�`�V�G�C�G�L�Y�`�m�y�(�4�A�F�M�R�V�R�M�A�4�(������� �(�5�A�G�N�Z�Z�_�Z�N�D�A�5�(�!�'�&�(�.�5�5�F�_�p�x�������������������x�_�S�F�:�9�F����������	�����������¿³²¶º¿���ػ����ûλû»����������������������������лܻ����� �����߻ܻл˻ûллллл�FFF$F'F$F"FFE�E�E�FFFFFFFFF�6�C�O�O�O�M�C�6�6�-�6�6�6�6�6�6�6�6�6�6 M F T 7 Y 8 ? & T  N 6 @ ) ' ( " 1 @ * ] H ) 8 r U ' Y V > I 0 $  Q U 2 8 H % E   F ' R M 0 & V \ K + * 9 q   B 9 \ 8  %  =  t  F  �  �  �  )  6  �  �  <  �  �  �  �  �  5  �  �  �  ]  9  ?  �  �  �  �  3  d    D  r  2  G  ?  �  g  �  �  �    �  I  �  �    �  V  �    t  �    �  W  �  �  G    BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM  BM    �  �  �  �  �  �  �  �  �  �  �  �  t  i  ^  R  G  ;  0                        
              �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  q  i  a  �  �  �  �  �  �  �  �  �  �  {  r  i  `  W  K  ?  2  &    �    <  A  F  J  N  S  W  [  Z  V  O  G  >  4       �  �  �    ^  �  �  �  �  ]  )    #  6    �  �  �  ;  �  �   �    "  4  ;  6  *        �  �  �  �  �  �  t  Q  ,    �  �  �  Y  �    `  �  �  �  g    �    �  �  
�  	�  �  g  Z              �  �  �  �  �  �  �  �  �  �  �  �  |  l  J  S  U  X  W  M  @  0      �  �  �  �  �  �  �  �  t  q              �  �  �  �  �  �  �  �  �  �  �  �  �  j  $  D  R  S  L  =  %    �  �  {  <  �  �  `    �  A  �  �  �  �  �  �    ,  2  2  .  &      �  �  �  s  ;  �  �    �  �  �  �  u  X  5    �  �  �    Y  3    �  �  �  �  �    #  9  Z  �  �  �  �  �  �  �  p  )  �  {    �    8  �  �  7  �  �    9  N  R  H  /    �  �  G  �  ;    �  �  p  �  �  �  �  �  �  �  �  �  �  x  f  T  A  )    �  �  z    	c  	�  
�    �  �  X  �  �      �  m  �    
�  
  i  r    �  �  �  r  ]  A    �  �  �  r  H  (    )  7  	  �  �  Y  L  R  Y  W  O  G  =  3  &    	  �  �  �  �  �  j  F     �        �  �  �  �  �  �  �  �  �  �  {  k  \  L  =  -    �  �  o  \  I  8  '    �  �  �  �  �  q  S  7     	  �  �  �  �  �  �  �  3  1  %    �  �  ?  �  �  >  �  �  T  �  P  �  	  	"  	  	  �  �  k  -  �  �  :  �  �  1  �    	  ]   z  �  �  �  �  �  �  �  �  �  ~  p  c  V  H  8  )       �   �  �  �  �  �  �  �    ]  ;    �  �  �  V  /        �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  k  �  �  �    t  i  ^  R  G  :  .  "      �  �  �  �  �  �  5  *  %  #  0  G  s  r  l  `  I     �  �  +  �  B  �  6   m  &                  �  �  �  �  �  �  s  D     �   �  A  �  �  �  �  �  �  f  ?    �  �  �  �  =  �  �  `  �    J  g  ~  �  �  �  �  �  �  �  �  �  v  X  9    �  �  r  �  �  �  �  �  �  x  `  E  .      �  �  �  �  �  c  $  �  �  �  �  �  �  �  �  /  �  �  '  2    �    �  o  �  +  )  	$  [  z  �  �  �    w  o  c  U  C  +    �  �  �  �  Z  :    e  V  H  :  +      �  �  �  �  �  c  :  �  �  �  H     �  �  �  �  �  �  �  �  �  �  d  C    �  �  �  �  [  c  "  Q  �  �  �  �  �  �  �  �  �  �  e  I  &    �  �  z  F     �  �  �  	  �  �  �  �  �  g  A    �  �  �  N    �  �  u  9       !        �  �  �  �  �  s  T  4    �  �  �  P    @  H  O  R  Q  I  =  '    �  �  �  ?  �  �  O  �  �  �   �  �  �  �  �  �  �  	2  	N  	G  	-  �  �  a    �    ~  �  �    �  s  f  Y  O  C  .    �  �  �  �  �  e  >    �  ~    g  �  �  �  �  �  �  �  �  �  {  Z  4  	  �  �  b  '  �  �  z  �  �  �  �  �  x  h  X  D  *    �  �  �  �  �  k  Q  7    �  �  �  �  �  �  l  U  ?  )    �  �  �  �  �    c  G  +  �  �  �  �  �  �  �  �  m  V  ;    �  �  �  }  ^  K  ;    3  D  <  3  )  &  !    �  �  �  �  �  o  9    �  �  C    �  �  y  m  _  R  C  5  %      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    t  h  \  P  D  8  ,  �  �  �  �  �  �  q  [  A  $    �  �  �  W    �  �  s  D  q  �  	.  	N  	V  	Y  	S  	>  	   �  �  X  �  r  �  j  �      �  2    �  �  �  �  �  �  �  l  M  &  �  �  f    �  /  �  j  �  �  �  �  �  �  t  c  Q  =  '    �  �  �  �  g  2  �  �  4  "      �  �  �  �  �  �  a  *  �  �  �  f  :        �  {  �  �  ~  q  ]  D  %  �  �  �  a  #  �  �  5  �  /  �    
    �  �  �  �  �  �  �  �  �  �  z  \  =    �  �  �      �  �  �  �  �  p  M  )    �  �  {  E    �  S  �  x  �  �  �  l  R  2    �  �  c  +  �  �  �  N    �  �  '  a  �  ~  \  9    �  �  �  �  l  O  3    �  �  �  �  �  �  