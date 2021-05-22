CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?��+J      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�1�   max       P��p      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��"�   max       <�t�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?333333   max       @F���
=q     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v}�Q�     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @O�           |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @��           �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��x�   max       <#�
      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B5�      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��    max       B4��      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >Rϑ   max       C���      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@�@   max       C�Ő      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          b      �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          G      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�1�   max       P�f>      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Q�   max       ?�����      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��"�   max       <�t�      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?333333   max       @F���
=q     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @v}�Q�     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @+         max       @O�           |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�X        max       @�@          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         A�   max         A�      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?ye+��a   max       ?�����     `  U�       	   	      	      b   B         1   '                                       	   4   %      *   	      
                  *         (                                          
                  	   O*kZN@	�O��O�[N��M�1�P��pPo1�N��pNe��O���P[OGP2�N���OW$8N�9N6�N�2�O�6�O�:!OO�v�NK�N�mqN�(>PJ�`Oyd�O�yO�9�O ��Nt��N�q�O�;BOM�O]�N�-�NR0P<1+N;�KO�O܁FNc��N�\jOLگN�,�OmWO�XN?qUN+^�N���OP�)O*�O3%�N���N��O���O��ANz�qNn�NN��N��N/�q<�t�<�t�<#�
;�o$�  $�  ���
���
�ě��t��49X�D���D���T���T���e`B�e`B�e`B��C���C����㼛�㼬1��9X��9X��j���ͼ���������/��/��/��`B��`B��`B���\)�t����������8Q�8Q�@��D���D���H�9�T���ixսm�h�m�h�m�h�q����+��\)��hs���-���T����\��"�����������������ABDOU[^[OB?;AAAAAAAA`aempz������}zyqmia`���������� ���������#),5<BJNOONB5*)$ $##�����������������������#0n�����{U<0
���������394*��������������������������2;?HSMMH=;;022222222��������������������u����������������sxu�������������������

���������� #$/BGHC?<7/#
����������������������������������������������������������������������������[bp{����������tg[X[ptx��������������tp��������������������|���������xw||||||||��������������������������������������������h[PE:0+6Bht������������� ��������mz�������������}vovm��*6CIRW]\RO*��@BJNP[\eglrrg[VNKFA@JOP[ahjjh^[XOLJJJJJJT[\^bhty~ytnh[TTTTTT"*5BNXgt���tgNB50)$"����������������������������������������')68>?=:6)(##''''''HOO[ag`[OHHHHHHHHHHH������
#
������IN[`fgrg[VNKIIIIIIII@HUanz������ztnaU<9@�����������������������
�����������45@BDHIJGCB@55444444GHPUakmlgdeaUSHC@@BG���������������������������������������� )6BOUWUOB6)$ ������������������������������������������������������������%)+5J[gppg]YONBA5)'%������������������������������������������������������������//1;<AHMUUYUH><<5///������
�������gt��������������tkcg����������������������������������������>IOUVWXXVUIGFDBB>>>>8<BHUWUQLHA<:45588888<HHNNH?<;9888888888�'�"�������'�4�@�G�M�T�X�M�G�@�4�'�U�R�U�X�a�k�n�w�z�r�n�a�U�U�U�U�U�U�U�UìéàÕÓÈÓàñù����������������ùì�4�(�!�(�4�A�M�Z�d�f�s���w�k�f�Z�M�A�4�����������������������������������������������������������������������������������������s�^�T�2� �1�s������������������������������ʼ��!�:�I�T�Y�P�G���ʼ��H�<�8�<�?�H�U�W�a�c�n�r�s�t�r�n�c�a�U�H����������������������������������������������4�@�M�Y�f�o�q�o�l�f�Y�M������ĵĬĿ�����#�<�U�b�o�m�t�n�0��������5�����(�A�Z�g�s�����������s�Z�N�A�5�g�d�[�V�[�\�g�t�t�g�g�g�g�g�g�<�1�/�#�����#�/�<�U�Y�a�g�g�Y�U�H�<�������������������������������������������������������������������������������������������������	��	���������׿y�y�m�Z�\�`�y���������ѿӿԿϿ��������y�������m�h�g�r�y�������Ŀݿ���ڿĿ�����������������	����'�4�2�)����y�f�Q�K�T�`�m�y���������ǿͿοʿ������y²¬²²¿������������¿²²²²²²²²ìæàÝÝààìù����������üùìììì�нνȽĽ��Ľͽнݽ��������ݽннн������	��+�)����������p�o�t�j�b�s�����׿ѿ˿������������Ŀѿݿ�����������ݿ�����������������������	��'�-�"�������ʿ.��	����������	��.�G�T�T�O�L�K�G�;�.��������*�-�6�C�O�U�T�O�G�C�6�*��e�Z�Y�M�Y�`�e�r�u�~���~�x�r�e�e�e�e�e�e�@�4�3�'�����'�3�9�@�H�K�@�@�@�@�@�@�׾ʾ����¾ʾ����	�������	����׾������ؾվԾ׾������"�)����	���.�*�.�6�<�E�T�`�m�x�������z�m�`�T�G�;�.�4�0�/�4�@�M�Y�f�q�g�f�Y�M�@�4�4�4�4�4�4�ù��������ùϹչܹϹùùùùùùùùùúɺ��������������ɺ��-�F�S�:�&�����ŔŋŒŔŠŨŭŮŰŭŦŠŔŔŔŔŔŔŔŔ�a�D�8�2�/�1�:�H�a�m�}���������������z�a���x�d�_�S�S�_�l�x�������ǻƻƻͻû������y�u�m�j�m�y�������������y�y�y�y�y�y�y�y����������������������������������������������������.�;�G�T�M�G�9�.�"��	���=�0�$����$�0�2�=�I�V�b�f�c�b�V�I�E�=�ּԼʼ����������ʼ˼ռ����������ּ�����������������������������������������
�����#�)�5�)���������������������������������������������������#� ���
�����
��!�#�0�4�6�0�0�#�#�#�#�a�X�U�L�<�.�*�/�<�E�U�a�n�{ÇÇÀ�r�n�a�3�3�7�=�L�Y�e�s�~�������~�z�r�e�Y�L�@�3�������������������ɺֺ������ֺɺ�����%�(�*�5�<�5�-�(����������F$FFFFFFFF$F$F1F5F;F=F>F=F1F&F$F$�l�`�J�G�=�G�N�`�l�y�������½��������y�lľĳĬĦĤĩħĳĿ��������������������ľ�û����ûлܻ���������ܻлûûûûû�����������������������������������򽫽��������Ľнݽ�����ݽнĽ���������E�ExE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�  J R h N 8 W ] S J J X 2 $ = 8 J V 8 } 6 > 4 + @ Z   9 ; ] $ X Q ? < G @ T _ 1 6 . i W ] V | @ @ S n c 9 D . T ` ^ 3  I C  g  l  i  �  +    �  �  )  v  K    Q  �  �  �  Y  �  �     t  �  c  �  �  �  �  !  �  7  v  �  �  �  �  �  Y  q  �      w  �  �  -  �  �  V  [  �    �  �  �  �  y  /  �  v  �  �  Z�e`B<#�
:�o�o�o�o���`��hs��1�T���q���P�`�#�
�����C����ͼ�C���9X�0 ż�h��j�C�������P�����P�y�#�'�C��\)�'�P�T���'Y��D���<j���-�8Q�aG����-�49X�T�������u��%�y�#�y�#�aG���o��\)��\)���
����������^5�Ƨ��T��E���E�������x�B��B��A�a�B�eBM�B5�B&�%B-_RB!�pA��B �B�[BVBF�BתB�BȗB�B�)B
��B GB*�6B�FB�B!��B �B\�BN]B/�;B�cB5\B�JBY~BmB��B��B9B"��B��B��B��B�AB�<B�-Bk�B�B}�B3]B!?�B�B"�B��B<BoB	�BK�B�@B)m�BE�B&��B��B�OB�!B�8A��BBBBRB4��B&��B.:wB!IgA�� B �B@�BAcB<�B��B ųB�(B�B�B	��BUB*�B�B�jB!�PB�<BA�Bi4B/��B�GB@!B3�B@&B��B<�B�/B<�B#<�B�B��B��B��B��B�]B�zB��B�eBB�B!T�B�B=1B��BCB>�B��BJ{B
AB)��B@B&�nB�AB��@��A��aA�I�A=��A��AK7�A���A��A�$A��@�QCA�F�A��%A�@A�V�B\
A���A�DAq��At�mA��Aph%A��A��:A*��A��A{2EA��A^փB �?�݂?�}�AWsAX�)Ah�8@��>Rϑ@I��A�E7A��@�٧An A��A]B�B
�jA a @��eA�e�ALtA� �AŊ�?�e�@7/�A�zgC���A`�A㜡@��IB
�A(C�4C�͇@��[A�b�A̝�A;��A���AK3�A�`�A	�A�o�A��@ջ7A��A���A�hA�u/B�"A��A���AombAt�+A�'�Ao��A���A�u�A+kHA��[A{�A�{�A]zB �y?��?��AV#SAW�;AiFh@�b>@�@@= �A�A���@�B=An��A��A\��B��A3@���A��AK�dA�}�A�W'?�Dv@;�9A���C�ŐA6A�~D@��B$�A(D_C��C�µ   !   	   
      	      b   C         1   (                  	                     	   4   %      +   	                        *         (                                          
            	      
                        G   =            3   '                  #   '      !            8      %   #            !               9      %   #                                                #                                    9   -            3                     #   '      !            %                                    +      %   !                                                #               N�B�N@	�O��O�[N��M�1�P�f>P�N��pNe��O-�P[OGO�T�N�g�N~��N�9N6�N�2�O�QtO�:!OO�v�N�<N�mqN�(>O�"[O:�;OQV�Oj��O ��Nt��N�q�O���OM�N�jjN�-�NR0P�N;�KO�O��XNc��N�\jO��N�,�OmWO�XN?qUN+^�N�\OP�)O*�O'��N���N��O���O��ANz�qNn�NN��N��N/�q    l  r  2  (  ^  �  |  E  ;  �  �  
    \  �  O    w     �  �  �  �  �  0  b  �  �  �  E  �  �  j  &  �  �  N  �  �  �  �  �  �  �  d  ]  �  :  �  �  �  �  m  �      �  ^  �  |  �<#�
<�t�<#�
;�o�o$�  �e`B�ě��ě��t��ě��D�����
�e`B�ě��e`B�e`B�e`B���㼋C����㼛�㼴9X��9X��9X�,1���+����/��/��/�o��`B��P���\)�49X������w���8Q�P�`�@��D���D���H�9�T���m�h�m�h�m�h�q���q����+��\)��hs���-���T����\��"���������������������ABDOU[^[OB?;AAAAAAAA`aempz������}zyqmia`���������� ���������#)5BGKGB5/))########���������������������#0b{������{U<0
��������
*-*$���������������������������2;?HSMMH=;;022222222��������������������u����������������sxu�������� ������������

����������#/354/#����������������������������������������������������������������������������[bp{����������tg[X[ptx��������������tp�����������������������������{y����������������������������������������46BO[nru~zsh[OLB844�����������������z|����������������zz*6>CIMNLE6*@BJNP[\eglrrg[VNKFA@JOP[ahjjh^[XOLJJJJJJT[\^bhty~ytnh[TTTTTT'*/5BNT[dt|�tgNB5-''����������������������������������������')68>?=:6)(##''''''HOO[ag`[OHHHHHHHHHHH�����

�������IN[`fgrg[VNKIIIIIIII@HUanz������ztnaU<9@�����������������������
�����������45@BDHIJGCB@55444444DHIU]ahjigda`UPHFBBD���������������������������������������� )6BOUWUOB6)$ ������������������������������������������������������������%)+5J[gppg]YONBA5)'%������������������������������������������������������������//1;<AHMUUYUH><<5///������
�������gt��������������tkcg����������������������������������������>IOUVWXXVUIGFDBB>>>>8<BHUWUQLHA<:45588888<HHNNH?<;9888888888������'�4�=�@�I�M�P�M�@�4�'�����U�R�U�X�a�k�n�w�z�r�n�a�U�U�U�U�U�U�U�UìéàÕÓÈÓàñù����������������ùì�4�(�!�(�4�A�M�Z�d�f�s���w�k�f�Z�M�A�4�������������������������������������������������������������������������������������d�Z�@�8�8�V�s�����������������������ʼ����üμ���!�.�:�H�I�F�?�!���ּ��H�<�8�<�?�H�U�W�a�c�n�r�s�t�r�n�c�a�U�H����������������������������������������'�&���"�'�4�K�M�Y�f�k�k�f�f�Y�M�@�4�'����ĵĬĿ�����#�<�U�b�o�m�t�n�0��������N�A�5�-�(�"��%�5�A�Z�g�s�������s�g�Z�N�g�e�[�X�[�^�g�t��t�g�g�g�g�g�g�H�A�<�1�/�<�H�U�\�[�U�N�H�H�H�H�H�H�H�H�������������������������������������������������������������������������������������������������	��	���������׿��y�m�]�`�y���������Ŀѿҿ˿Ŀ����������������m�h�g�r�y�������Ŀݿ���ڿĿ�����������������	����'�4�2�)����y�f�Q�K�T�`�m�y���������ǿͿοʿ������y²°²¶¿������������¿²²²²²²²²ìæàÝÝààìù����������üùìììì�нνȽĽ��Ľͽнݽ��������ݽннн������~����������������	���������������������������Ŀѿݿ����������ݿѿĿ���������������������������������������"��	��������� �	��"�.�;�B�C�B�@�;�.�"��������*�-�6�C�O�U�T�O�G�C�6�*��e�Z�Y�M�Y�`�e�r�u�~���~�x�r�e�e�e�e�e�e�@�4�3�'�����'�3�9�@�H�K�@�@�@�@�@�@��׾ʾþ¾ƾʾԾ�����	�����	�����������ؾվԾ׾������"�)����	���T�H�G�B�D�G�O�T�`�h�m�y��{�y�m�`�^�T�T�4�0�/�4�@�M�Y�f�q�g�f�Y�M�@�4�4�4�4�4�4�ù��������ùϹչܹϹùùùùùùùùùú����ź��������ɺ���!�0�.�&�����⺽ŔŋŒŔŠŨŭŮŰŭŦŠŔŔŔŔŔŔŔŔ�a�D�8�2�/�1�:�H�a�m�}���������������z�a���x�e�_�^�l�x���������ǻǻŻƻ˻û������y�u�m�j�m�y�������������y�y�y�y�y�y�y�y�����������������������������������������	����������	��"�.�9�;�B�;�3�.�"��	�=�0�$����$�0�2�=�I�V�b�f�c�b�V�I�E�=�ּԼʼ����������ʼ˼ռ����������ּ�����������������������������������������
�����#�)�5�)���������������������������������������������������#�#���
���
���#�0�2�4�0�'�#�#�#�#�a�X�U�L�<�.�*�/�<�E�U�a�n�{ÇÇÀ�r�n�a�3�3�7�=�L�Y�e�s�~�������~�z�r�e�Y�L�@�3���������������ɺֺ�������ۺֺɺ�����%�(�*�5�<�5�-�(����������F$FFFFFFFF$F$F1F5F;F=F>F=F1F&F$F$�l�`�J�G�=�G�N�`�l�y�������½��������y�lľĳĬĦĤĩħĳĿ��������������������ľ�û����ûлܻ���������ܻлûûûûû�����������������������������������򽫽��������Ľнݽ�����ݽнĽ���������E�ExE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D� % J R h ; 8 G [ S J 9 X 0 ! 5 8 J V 6 } 6 > > + @ ]  ? 5 ] $ X J ? ) G @ D _ 1 2 . i E ] V | @ @ O n c 5 D . T ` ^ 3  I C  �  l  i  �  �    �  �  )  v  �    Q  �  �  �  Y  �  �     t  �  G  �  �  C  �  �  �  7  v  �  a  �  �  �  Y  �  �    �  w  �  N  -  �  �  V  [  �    �  q  �  �  y  /  �  v  �  �  Z  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  A�  |  �  �  �        �  �  �  �  q  F    �  �  =  �  �  �  l  w  �  �  �  �  �  �  �  �  ~  z  u  n  f  E    �  �  k  r  k  d  `  \  Z  Y  Z  \  _  b  b  ^  Z  U  Q  P  P  `  p  2  /  ,  (        �  �  �  �  �  �  �  �  i  N  3     �          $  &        	        �  �  �        #  ^  ]  \  [  Z  X  W  V  U  T  T  S  R  Q  P  P  O  N  M  M  <  �  �  �  |  }  w  e  F    �  �  )  �  ]  �  Z  �  �  Q  >  l  w  w  v  y  |  s  `  ?  	  �  x    �  F  �  +  b  �  E  5  #      �  �  �  �  q  <     �    @  �  �  2  �  I  ;  5  /  )  #        �  �  �  �  �  �  �  }  l  [  K  :    g  �  �  �  �  �  �  z  ?  �  �  2  �  >  �  �  �  �  *  �  �  �    d  k  O  @  Q  P  G  4    
  �  �  <    �  S  �  �  �  �    	    �  �  �  �  �  �  X    �  i    �  {          	    �  �  �  �  �  �  �  r  V  8    �  �  �      "  /  7  9  5  2  2  D  [  N  8    �  �  L    �  �  �  �  �  �  �  �  �  �  �  x  \  =    �  �  �  �  j  ?    O  J  E  ?  :  5  0  4  =  F  O  X  a  d  [  R  I  @  7  .            �  �  �  �  �  �  �  j  K  ,  
  �  �    ;  k  t  u  n  b  T  D  5  $    �  �  �  �  _    �  c  �   �       	      �  �  �  �  �  }  a  ;    �  �  �  2   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  a  G  0  "        �  �  �  �  �  �  T    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  e  Q  =  �  �  �  p  [  E  &    �  �  �  g  =    �  �  �  y  L    �  �  �  �  �  �    p  ^  K  6      �  �  �  �  �  g  6  �  �      +  .  -  0  &  	  �  �  m  &  �  n  �  G  �  a  <  O  ^  b  W  B  *    �  �  �  ^    �  m    �  %  �  ,  a  j  v  ~    }  z  �  �  �  �  �  m  H    �  �  M   �   �    :  V  j  v  ~    w  f  E    �  �  y  3  �  e  �    %  �  �  �  �  �  �  �  �  �  �  �  z  g  F  %    �  �  �  �  E  C  @  ;  1      �  �  �  z  S  %  �  �  �  h  5    �  �  �  �  ~  c  F  I  �  �  �  �  �  [  %  �  �  q  +  �  �    �  �  �  �  �  �  q  S  *  �  �  �  �  @  �  �  _  ,  �  j  \  L  9         �  �  �  �  �  �  �  y  Z  5     �  �  �  �         $  %  %      �  �    M    �  �  b  �  U  �  p  ]  F  +    �  �  �  f  7  	  �  �  p  4  �  �  _  X  �  �          �  �  �  �  �  �  �  ^  ,  �  �  Z    �  �    :  L  L  6      �  �  w  =  �  �  R  �  Q  �  �    �  �  �  �  �  �  �  �  �  i  8    �  �  g  .   �   �   |   >  �  �  w  a  Q  =  %    /  %        �  �  �  �  N    �  �  �  �  �  �  �  l  @          �  �  �    �    �  �  �  �  �  w  g  W  F  0    �  �  �  �  r  M  )     �   �   �  �  �  �  v  ^  F  )  	  �  �  �  �  e  I  7  2  .  2  ;  C  �  �  �  �  �  �  �  �  r  L    �  �    �    o  �  Z  �  �  �  �  {  _  Q  T  z  �  �  �  �  �  �  n  O  .        d  _  Q  ?  -    �  �  �  �  �  �  k  W  B  %  �  �  x  >  ]  K  K  Z  H  2    �  �  �  q  W  <    �  �  �  m  @    �  �  �  �  �  �  v  T  .  �  �  �  `  )  �  �  {  >  �  �  :  3  ,  $          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  j  i  j  g  R  =  '    �  �  �  �  �  �  �  {  k  \  L  9  %       �  �  �  �  �  �  �  �     �  �  |  [  9    �  �  �  �  �  |  ^  ?      �  �  �  �  �  �  �  �  �  �  �  {  d  J  -  	  �  �  h    �  I  �  u  m  =  
  �  �  y  M     �  �  �  `  1  	  �  �  �  <  �  }  �  i  P  .  
  �  �  �  �  ^  <    �  �  �  �  a  ;    �      	     �  �  �  �  �  �  �  �  �  m  J    �  }  %  �    �  �  c  <  0  2  E  (  4    �  �  �  g  -  �  �  3  �  �  �  �  �  �  �  �  y  m  a  Y  W  V  T  R  N  K  G  D  @  ^  =    �  �  �  y  K    �  �  �  h  ?    �  �  �  �  �  �  �  �  �  p  ]  K  4      �  �  �  �  u  [  A  (    �  |  \  =  -          �  �  �  �  �  �  �  h  G  '    �  �  �  �  e  Q  <  *      �  �  �  �  �  �  �  �  �  �  