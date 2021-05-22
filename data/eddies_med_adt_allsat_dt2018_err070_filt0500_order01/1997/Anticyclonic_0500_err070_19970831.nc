CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�V�t�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       Nv�   max       P���      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ���   max       >�      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @FK��Q�     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?���R    max       @vl(�\     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @)         max       @N@           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�@        max       @��          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��C�   max       >�J      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B1Y�      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B1��      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?Y�H   max       C�	W      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?T�'   max       C��      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          =      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       Nv�   max       P��      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z����   max       ?��~���%      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       >I�      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�z�H   max       @FK��Q�     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?ᙙ���    max       @vi�����     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @N@           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�@        max       @���          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D   max         D      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?}��,<�   max       ?���"��a     0  O�         [      
   
               $            �         E   6      
            ;      m   8   -            D      ;                      )      �   �   0   �         p   
         +   O�,NvtP���O|cqNc�N�%�O%~�N�_N�4NP�DO�YO��JO>ցOAJP��O �?Nv\SPO��P
y?O�vN���O���OI�NQ�P���O*CP�� P��Oɍ7N�C1Nv�N�F�PJ%N�'O��N�,�O>t�ODz$N�r>OT�!O���O� �Ob.�P �O��O�/PT3OӦO
��O��vN���NfIN/:On�wN�Q���㼛�㻣�
:�o:�o;D��;�o;�o<o<#�
<#�
<T��<e`B<�o<�C�<�t�<���<���<���<���<��
<��
<�1<�9X<�j<���<�/=+=+=+=C�=C�=t�=��=#�
=#�
=0 �=0 �=0 �=<j=<j=P�`=T��=T��=ix�=y�#=�o=�\)=�\)=���=�{=���=�"�=�`B>�ddegjt��������tnhgdd��������������������������5QZ_NB)����pmu��������������|wptst���������ttttttt��������

�����������������������#%0<<=IQUUUUI<0$#�������������������������������������������������������}uxv���������������}���������������������������
$&��� )B[t������[B)��������������������!"/;>>;9/#"�������� ���������������0;B@5)�����(%&,/6<HU]ceed_UH</(������	

���������XZgt������������tg[X����67BIEB<6����������������������������)3XWC�������������������������\]ez�������������nb\BCI[h�����������h[NB������
#,..,#
����
##,/)#
����������������������������������������4;Oh�������{wh[OB1)4]bfns{���������{nb]]YVX]am���������}zmaY2-5;BN[ae[ZNB5222222�����������������������
#./32/-#
����9?CJOX\hikhd\OCC9999		
#+/1/*'# 
""#'/<@OY^^\\UH</'"��������*2������ZTUX^amz��������zmaZ���������������������������

����������-5:<;5���z|������ 	������z���� yuzz�������������zyy�������
%' 
���������������������������������������������../8<=BEF<7/........$/;<FE><8/#FEBEHU``_YUHFFFFFFFF�zÇÓÜàææàÓÇ�z�n�a�V�Z�a�n�q�z�z�����������������������������������������<�U�nŇŠůŻŸŔ�}�n�b����������#�<����3�.�,�'�������ܹ̹Ϲܹ�����L�Y�\�e�g�e�\�Y�X�L�@�9�@�J�L�L�L�L�L�L�r���������������������}�s�r�g�r�r�r�r����)�4�5�?�5�*�)�������������� ����������������������������~�|�����������������(���������ݿڿݿ����a�n�n�w�s�n�a�[�U�]�a�a�a�a�a�a�a�a�a�a�������ûܻ��߻һû��������������������	�"�.�;�T�W�`�k�`�G�;�"����;ʾľ;�	�������ͼּڼ��ּʼ��������������������'�3�@�L�R�Y�a�d�Y�L�H�9�3�'������'����B�hāĎĤĦĚā�[�6�3�4�)���������'�4�>�@�C�D�@�6�4�'�$����	�����/�;�B�@�B�;�/�&�"��"�$�/�/�/�/�/�/�/�/àì��������������ùÓ�z�U�=�9�@�U�aÇà�������������������������x�w�~�����������T�a�m�����������������z�m�a�T�G�?�<�?�T�#�/�<�?�B�G�<�8�/�#�!�����#�#�#�#�#������������������������y�q�t�y�{�|����ʼּۼ������޼ּʼü�����������������������������������������������������������	��&�5�4�%�	��������������������Óàìùþ��ûìàÓÍÇ�}�z�x�w�~ÇÑÓ���������������������g�Z�W�J�J�^�k�s���s�����������������f�Z�A�9�>�A�E�G�M�Z�s��(�4�A�I�M�_�c�e�Z�M�A�4����������������	�������߼������������������(�*�(����������������/�<�H�U�U�U�T�H�>�<�/�,�'�+�/�/�/�/�/�/�'�4�D�S�Y�T�M�@�4����ܻ׻ڻٻݻ���'��� �����!�#�%�!�����������������čĚĦĳ������ĿĳĦčā�q�j�h�[�T�c�~č�ݿ��������������ݿٿӿٿݿݿݿݿݿݿ������ĿѿӿͿǿ������������������������������������y�w�m�`�X�T�L�L�P�T�`�m�y�����	������
�	������������������ʾ׾�	����	�����׾ʾ����������žʾ4�M�Z�s����x�s�f�Z�A�4�(����
��(�4�����������������������l�`�^�X�V�X�g�l��ŔŠŭŹ��������������ŹŭŧŠŔŋňŌŔ�����ֻ�� �!��
����ֺȺ�������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�DyDuDzD�D�D��������
���	����������¸²¿���ؼ4�M�i�w�w�n�Y�?����ܻ˻ʻϻۻ��'�4�uƁƎƚƧƳƷƿ������ƳƧƚƓƁ�u�n�r�u���
����������
����������������EPE\EiE�E�E�E�E�E�E�E�E�EiEaE]E[EVELEHEP�Z�g�k�s���������������s�m�g�Z�Z�V�W�Z�Z�����ĽȽнֽнĽ��������������������������"�*�,�*������������������������#�0�0�#��
�������������������ŭŹ������������ŹŭŨţŭŭŭŭŭŭŭŭ S y 6 8 : 4 / H < L % d I < H ( 6 8 O $ ! P U I H M 1 J 5 V s 3 < h % O J J @ x J e  < * 6 Y Y W F H l G J I    J  x  �     @  �  f      �  e  N  �  �  �    r  �  �    �  ;  �  Q    �  �  �  �  �  O  �  �  6  1  �  �  �  �  %  �  �  �    �    	  z  s  �  &  \  L    ���C��T��=� �<��<#�
<T��<e`B<D��<�j<�C�=<j=+=0 �=o>�J=C�<�9X=�9X=�t�=0 �<�h<��=<j<���=��=aG�>
=q=�E�=��-=#�
=��=@�=���=ix�=ȴ9=<j=ix�=�t�=P�`=��=��w=�j=���><j>Xb=�;d>aG�=���=�E�>F��=\=��=>�R>n�B	��B"�B#%BSvB�B#�nB�B&$�B�:BT�B"��B�B Y'B�OBjTB!�A���BfB��BFB�B
�)B��Bo�B޵B"�Bi�Bu�B#�ZB$��B��B��B��B(�`B �_B�B��BhB1Y�BY�B�cB,��A��B(�B�B�B4CB��BiUBZeBBo�B��B��B�B	��B"@B��BA B�uB#��B�4B&B�B�_BB�B"�TB�
B ?B~/B@�B!�0A���B�pB��B?{B2B
�0B��B��B��B"?�B�3BFcB$A�B%9�B@�B��B��B(�DB �%B:B�B=�B1��Be)B��B,G]A��B?�B<rB��B��B�!Bf�B��B�B@�B�&B��B��AȢp@]A���?Y�H?�B@�/�A��
@�@�A�A��B@�ӍA\�b@�H{?��3Aږ�@���A���AʢA�+%A�hbA�vAHv@��MA��A�_�A�G�A�1AD;AA7��A�>A�`�Aí�@�/@[�LA�Q�A~�At�Ak��AY=�ATP&A;e}A,\A���@F'�C��YA�А@��BB�A��C�	WA��qA%��A��_A�PA�$�Aȇ�@��A쀼?T�'?�$@��A�e�@��(A��zA� b@��dA[4�@���?��kAڀP@ƩwA��A���A�|tA�s�ADAG:@���A���A�k�A�s�A�s�AE�A6��A�A�~A�~�@���@[�FA߂�AxAshaAj�AY
�AW�YA;�iA�A��Q@I��C���A�}�@��BH�A���C��A��QA$�	A���A�xIA��5         \      
         	         %            �         F   6      
            <      m   9   -            E      ;            	      !   *      �   �   1   �         p            ,            ;                           )         =         /   %                  ;      7   )   !            )      #                     %      +      %   4         '                        '                           !                                       9      '      !            !                                          '                        N��NvtP��N�/Nc�N'�O%~�N�_N���NP�DOLS�O���Nߖ�OAJO���O �?Nv\SO�S|O�DO�vN���O���OI�NQ�P��O
�/P(Q�O���O��IN�C1Nv�N�F�O�=N��Oj@�N�,�O�O
�RN�r>OT�!O�X�N��Ob.�O���O���O��BP,�OӦO
��O��N���NfIN/:O��N�Q�    �    �  �  �  a  \  ,  �  �  �  O      �  �  �  �  }  I  '  H  �  �  F  �  �  �  �  �  d  7  �  �  N  �  �  �  q  �  �  �  -  ,  8  t  �  A  �    K    �  �+����<�9X<D��:�o;ě�;�o;�o<D��<#�
<��
<u<�1<�o>I�<�t�<���=49X=t�<���<��
<��
<�1<�9X<�/<�`B=}�=T��=C�=+=C�=C�=D��=#�
=u=#�
=8Q�=H�9=0 �=<j=D��=���=T��=ȴ9=�\)=�\)=���=�\)=�\)=�h=�{=���=�"�=��m>�gfgjt�������trlggggg�����������������������)BIPQMB5)����~yv���������������~~tst���������ttttttt������

��������������������������#%0<<=IQUUUUI<0$#��������������������������������������������������������wzy������������������������������������������
$&���/+*,5BN[gv||yrg[NB5/��������������������!"/;>>;9/#"������������������������(0443)����(%&,/6<HU]ceed_UH</(������	

���������XZgt������������tg[X����67BIEB<6����������������������������)1=TQB������������������������vqv���������������zvNLOX[h����������th[N������
#+-.,#
����
##,/)#
����������������������������������������88<CR[t�������th[OB8`bhnu{���������{nb``llsz������������zrnl2-5;BN[ae[ZNB5222222������������������������
#*/0/,%#
��9?CJOX\hikhd\OCC9999		
#+/1/*'# 
! $%/<>MW]\[ZUH</($!��������������������ZTUX^amz��������zmaZ��������������������������� 

�����������&1685)�������������������������� yuzz�������������zyy��������
 
��������������������������������������������../8<=BEF<7/........#(/5;<84/*#FEBEHU``_YUHFFFFFFFF�zÇÌÓßÕÓÇ�z�n�a�\�_�a�n�z�z�z�z�z�����������������������������������������I�c�r�}��s�]�U�<�#��������#�0�I�������"� ������������������L�Y�\�e�g�e�\�Y�X�L�@�9�@�J�L�L�L�L�L�L�������������������w�x������������)�4�5�?�5�*�)�������������� ����������������������������~�|��������������������������������������a�n�n�w�s�n�a�[�U�]�a�a�a�a�a�a�a�a�a�a�������ûӻ׻Իлɻû��������������������	�"�.�;�F�S�Z�T�G�;�2�"����Ѿ;վ��	���������żʼмԼμʼ��������������������'�3�@�L�R�Y�a�d�Y�L�H�9�3�'������'�B�O�[�h�yăĈĈă�t�h�[�O�B�9�5�2�4�9�B��'�4�>�@�C�D�@�6�4�'�$����	�����/�;�B�@�B�;�/�&�"��"�$�/�/�/�/�/�/�/�/Óàìù��������ùìàÓ�z�c�W�\�d�n�zÓ�����������������������������������������T�a�m�����������������z�m�a�T�G�?�<�?�T�#�/�<�?�B�G�<�8�/�#�!�����#�#�#�#�#������������������������y�q�t�y�{�|����ʼּۼ������޼ּʼü��������������������������������������������������������������#�1�0� �	��������������������àìùû��ùùìãàÓÇ�|�zÁÇÓØàà�����������������������������v�n�g�j�|���s�������������������s�f�Z�U�S�T�U�^�s��(�4�A�I�M�^�c�e�Z�M�A�4����������������	�������߼������������������(�*�(����������������/�<�H�U�U�U�T�H�>�<�/�,�'�+�/�/�/�/�/�/����'�4�@�F�R�R�M�@�4�������������������!�!�"�!������������������ĚĦĳĸĻĸĴĳĦĚčĄā�{�w�t�xāčĚ�ݿ��������������ݿٿӿٿݿݿݿݿݿݿ����ÿĿȿſĿ��������������������������m�y���������~�y�n�m�j�`�T�S�P�T�U�`�g�m���	������
�	������������������ʾ׾�	����	�����׾ʾ����������žʾA�M�Z�s�~�v�s�f�Z�A�4�(���	���(�4�A�y�����������������y�o�n�o�v�y�y�y�y�y�yŔŠŭŹ��������������ŹŭŧŠŔŋňŌŔ�ֺ������������ֺɺƺ��������ɺ�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}DxD}D�D�D�������������	�	�����������±§¬º���˼'�@�M�Y�`�^�@�4�������߻ݻ�����'�uƁƎƚƧƳƷƿ������ƳƧƚƓƁ�u�n�r�u���
����������
����������������EuE�E�E�E�E�E�E�E�E�E�E�E�EuEjEeEfEiEpEu�Z�g�k�s���������������s�m�g�Z�Z�V�W�Z�Z�����ĽȽнֽнĽ��������������������������"�*�,�*������������������������
�����
��������������������ŭŹ������������ŹŭŨţŭŭŭŭŭŭŭŭ Q y )  : 4 / H = L  b D <  ( 6 ' / $ ! P U I ? K * 5 5 V s 3 < e  O C ? @ x E N   + $ R Y W % H l G = I    �  x  �  �  @  B  f    �  �  �  �    �  ^    r  �  %    �  ;  �  Q  �  .  �  e  �  �  O  �  �    �  �  Q  G  �  %  E  �  �  A  >  V  �  z  s    &  \  L  c  �  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  D  �  �              �  �  �  �  �  �  o  3  �  Y  �  g  �  �  �  �  �  �  z  w  v  t  k  Y  G  4  !    �  �  �  �  �  F  �  �          �  �  �  �  �  �  �  -  �  �  �  >  �  /  �  �  �  �  �  �  �  �  �  �  �  ]  $  �  �    �  �  �    =  <  8  1  )      �  �  �  �  �  o  K  &     �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    a  `  _  ^  ]  Y  U  Q  P  T  T  Q  L  D  8  *    �  �  Y  \  X  U  Q  M  H  A  :  1  '        �  �  �  �  �  �  �  �    "  )  *  ,  ,  (      �  �  }  =  �  �  k    �  `  �  �  �  �  	    %  4  B  P  _  m  z  �  �  �  �  �  �  �  X  �  �  �  �  �  �  �  �  �  k  5  �  �  w  ,  �  h  �    {  �  �  |  l  Z  C  2  %    �  �  �  �  �  n  ;  �  U   �  �    +  A  N  O  L  C  =  5  +      �  �  V  �  �  !  �    �  �  �  �  �  �  �  �  �  �  o  P  -    �  �  m  2      ;  v  t  A  �  �    T  z  z  9  �  �  �  h  �  }  	^  �  �  �  �  �  s  d  S  ?  (    �  �  �  m  G    �  �  X    �  �  �  �  �  �  �  �  x  m  c  X  M  B  5  '          �  �  $  A  Y  p  �  �  �  �  �  ]  '  �  �  F  �  E  t  |  �    E  h  �  �  �  �  �  �  �  �  j  +  �  l    �    w  �  }  {  z  z  v  i  X  C  (    �  �  �  E    �  \    �  �  I  F  D  6  '      �  �  �  �  �  k  N  3    �  �  �  C  '      �  �  �  �  �  x  X  6    �  �  �  y  Z  >  (    H  A  2    �  �  �  �  k  >    �  v  (    �  �  C  �  �  �  �  �  �  �  s  h  \  P  D  9  -  !    
  �  �  �  �  �  �  �  �  �  �  p  H    �  �  Y    �  E  �  �  a  &  �  '  �  &  E  <  -      �  �  �  y  d  )  �  p  �  k  �  G  �  �  &  H  f  �  �  �  �  �    3  �  j  �  d  �  <  �  �  �  �    0    �  �  �  �  �  �  f  0  �  �    �  �  ,     I  �  �  �  �  �  �  �  k  L  &  �  �  �  *  �  ]  �  @  t   �  �  �  �  �  |  h  S  ?  *      �                �  �  �  �  �  �  �  o  Z  F  1    �  �  �  y  P  '   �   �   �  d  a  ]  W  L  A  4  '      �  �  �  K  	  �  x  (  �    �    ,  6  5  /  !    �  �  �  �  g  9  �  z  �  "  �  �  �  �  �  �  �  w  a  I  -  	  �  �  }  /  �  �  =  �  +  �  y  �  "  ^  �  �  �  �  �  �  p  ;  �  �  T  �  q  �  �  �  N  F  >  7  (      �  �  �  �  �  �  �  �  �  d  X  K  ?  �  �  �  �  �  �  �  �  �  v  e  Q  7    �  �  W    �  ]  �  �  �  �  �  �  �  �  a  2  �  �  r    �  I  �  z  �  l  �  �  �  �  x  k  Y  G  3    	  �  �  �  �  |  O     �   �  q  P  )  6  5  #  	  �  �  �  �  �  p  L  !  �  �  �  =  m  �  �  �  �  �  �  �  �  �  t  k  J  "  �  �  O  �  {  �   �    �  �  �      /  9  Q  j  �  �  �  �  i  +  �  Y  `  }  �  �  �  �  �  �  i  I  &  �  �  �  c  %  �  �  ?  �  �   �  �  �    m  �  �  "  )    �  u    �    j  
�  	�    C  �    *  )      �  �  �  �  Y    �    @  -  �  �  �  P  �  '  '    0  2  !  
  �  �  �  [     �  q     �  	  u  �  ~  �  �  $  L  j  s  d  >    �  V  �  7  g  w  �  
h  	  �  �  �  o  ^  O  X  G  &    �  �  �  q  J  %  �  �  �  �  �    A  $          �  �  �  �  �  �  P    �  U  �  �  5  �  6  l    R  ~  �  �  ~  \  '  �  o  �  Q  {  [  
   �  ]  �      �  �  �  �  �  �  �  �  p  R  4    �  �  �  �  �  �  K  5    	  �  �  �  �  �  ~  J  �  �  k  9    �  �  _  $    �  �  �  �  �  �    X  *  �  �  �  o  >    �  �  s  =  �    9  _  �  �  �  u  M    �  �  (  �    �  �  "  [  �  �  �  �  �    c  E  $    �  �  v  M  7  !    �  �  k  �