CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�
=p��
      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P</�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ����   max       =��      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @E������     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @vD          �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @(         max       @Q            t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�d        max       @�^�          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >m�h      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,p�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~�   max       B,T8      �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?���   max       C��+      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?���   max       C�ק      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          1      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          %      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P��      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��*�0�   max       ?⒣S&      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ����   max       >+      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�ffffg   max       @E�=p��
     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�(�\    max       @v@(�\     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @'         max       @Q            t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�d        max       @��@          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?@   max         ?@      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Fs����   max       ?���$tT     �  QX      	      
   
         0   *            
         2   K               
         )   j            4         D         +         
            �         	         �         &         +      YN�'�N��5N�(CN �OO�̇Nu�OډP,�PNͶYNf�N#�N�i�NBN�NH�BO��CP;RN1&�OQNNș�O#��Oa!EN��O#�O���P</�O���N�~FO�r&P7
�OyeSO��~P!�lO-D�N�GO�g�N��N rN��P.��OFU�N��>O��fNt��Nu�N�v�O��Oj�O��N#�N��O/�
N:_�O#gO��M��O��ڼ���������j�e`B�#�
�o�o�ě����
;o;D��;�o;��
;ě�<o<o<t�<t�<#�
<#�
<D��<�o<�t�<�t�<���<�1<�9X<�9X<�j<�j<ě�<ě�<ě�<ě�<�/<�`B<�h<�h<��=�P='�=,1=,1=0 �=0 �=0 �=D��=Y�=Y�=]/=]/=q��=q��=��=���=���=��!)---*)nrrt����������wtnnnn20--15?BDBBCFB@52222\[Y\ainxz}{zwpnkda\\UV\bhnx{}���|{ngbYUU�����
%/7<;=<#
���3+268BLDB63333333333�������
�������sru}��������������zs#037420-#����������������������������������������\`gt������tthg\\\\\\����������������������������������������edfn�����������txnhe	
)<NYaa\[]N)///<<HNLHA</////////�������
"
����|{|�����������wutz�������������}zw�����#+581)�����)-00)��������������������#%$)/<Hdmh`QKH<62)%#��������)9=(�����&$')5BN[gllnng[B:1)&#/<HHIHE</#"/8LVXVH;/%"!a]_mt�������������ta )5BCGKNV[f[B5)+059?@BNTV^a`^[N5)ht����������������th#0<IUYRID:0*#�������������������������
#/H]`LG@<0#�����������������������zy}���������zzzzzzzz�������	��������rpnos{�������������r������
$*#
������������������������)6<BJUW[d[QB6)����
��������� "�������������������������#))&���������(+,)$���������
 
��������������������������������

��������ztv{���������������z��������������������;<@HJTUamnqomlaTH@;;)6BEONKIGB86)5336>BDCB65555555555�z~�����������������ù������������������ùìêãìöùùùù��������������������������������������������)�6�<�B�G�B�6�)��������������	�����	�������׾˾׾���𻓻����������������������x�v�u�x�|�������`�m�y���������������y�m�`�Q�@�E�K�V�Y�`�zÇÓÕÓÌÇ�z�y�s�z�z�z�z�z�z�z�z�z�z�(�4�M�f�t�����s�f�Z�A�(����������(�s�������������������������l�b�Z�W�Y�a�s�-�:�F�M�S�Y�Y�S�F�:�-�%�!��!�"�-�-�-�-����*�3�*�*�����������������'�(�1�2�'���������������uƁƇƍƃƁ�u�h�\�W�\�\�h�k�u�u�u�u�u�u�����������������������������������������h�t�w�t�i�h�^�[�O�K�O�P�[�]�h�h�h�h�h�h�Ľݽ���	������н��������{�������������	��H�T�m�l�a�Z�H�/�"�	���������������z�����������������z�s�x�z�z�z�z�z�z�z�z�	��/�;�H�N�Q�T�]�T�M�H�;�/�"�����	�f�s�����������������s�f�f�`�c�f�f�f�fčĚĦĨıĲĳĳĦĥĚčąā�~�|�~āĆč�ݿ��������������ѿݿ��ݿ�ֿ�ƁƎƔƔƎƁ�u�l�u�}ƁƁƁƁƁƁƁƁƁƁ�4�A�M�W�Z�`�f�s�~�s�f�Z�M�A�4�(���(�4��������������������ùìù�������뻑�����ֻ޻ݻӻѻֻݻܻл����r�b�]�j�x���`�m�y���������~�|�y�`�T�G�;�6�4�;�G�T�`�zÇËÓÖÛÝÓÓÇÀ�z�v�x�z�t�z�z�z�z�H�T�a�d�d�a�T�H�;�/���������������/�H�A�N�g�������������s�g�Z�N�E�L�@�=�,�$�A¦¿������²¦�z�u�x�}��������$�I�V�[�Z�I�0�����������������t�~��x�g�[�A�)�����������������B�l�t�Y�f�r���������������t�f�Y�O�M�B�M�R�YÓØàãìùúóìà×ÓÈÇÁÇÐÒÊÓ���5�<�@�F�P�a�g�Z�N�5�(����������Z�_�f�s�|���������t�s�f�]�Z�M�I�M�X�Z�׾����� �������׾о׾׾׾׾׾׾׾��<�H�U�_�a�n�n�s�n�a�U�O�H�?�<�:�<�<�<�<Óì�����������
��������ìàÐÈÃÁÓ���������ʾѾɾ���������������������������'�4�@�K�M�U�M�@�4�'����������f�������������f�M�8�4�1�'�&�)�0�>�M�f�"�.�;�A�?�;�.�"��	��	���"�"�"�"�"�"�ʾ׾�����׾ʾʾɾǾʾʾʾʾʾʾʾʽy�������������������{�y�p�l�`�\�`�l�w�y�	��"�/�6�7�3�/�+�"���	������������	�#�0�<�I�U�b�i�n�x�n�b�U�I�<�0�#����#D�D�D�D�D�D�D�D�D�D�D�D�D�D{DnDgDiDoD{D��/�/�#�����#�/�3�<�<�<�/�/�/�/�/�/�/��#�/�<�=�=�<�/�*�#������������'�4�@�F�K�H�@�4�'����������������� �����������������b�n�{ŅŇœŔśŗŔŇ�{�s�n�a�U�Q�U�`�b�e�r�������ºкպɺ��������~�o�Y�V�P�W�e�������������ٺ����������⼘�����ʼ�����ּʼ����������������� > J n M  $ R 9 $ ( e J > Z z < J Z I H $ � Y h 5 N , Z Z 0 P � m U \ S L � T $ R T ? X * 7 ' Q  S  , N J U W 4    �  �  �  �  &  U    �  �  �  �  R  �  �  �  A  T  `  �  �  f  }  *  �  7  d  K  &  '  1  	    �  �  �  e  1  `  �  �  �  �  �  �  �  �  Q  �  �  `  �  x  c  y  }    ����
��t����㻃o%`  <�9X��o=,1=�P<�C�;�`B<t�<e`B<e`B<D��=ix�=���<49X<�9X<�9X<�<���<�9X=o=q��>J=8Q�=+=Y�=��P=t�=��=�^5=�P=�w=�\)=�P=o='�>&�y=e`B=m�h>2-=H�9=<j=T��=m�h=�7L>m�h=�+=�7L=ě�=�hs=�9X>   =�->;dZBb4B
BNB�eB��B(?IBD9B�B#,;B~�B%@B��B�aB	ΦBx�B�LB<�B3�B��B
�B�B �.BHwB�B��B��B�HB�B,CA���B
քBi�Be�BlEB%�ZB!�B��B��B�eB��B�PB2-B��B��B�/B3B,p�BB�bB�B!�BFBU�B!\�A��
B�dB�B`B?lB
3fB'.B��B(?-B?B��B#?�B@BB%B�B��B��B	��B��B�IB@UBC/B94B�)B?�B ��B@YB�CB@)B�VB@�B��B��A�~�B
�RB�BE�B�B%�$B"�B�B@�B�'BҼB��B?�B?�BCiBS�B6zB,T8BGqB�!B=�B?�B�2B=B!�A�|[BW"B(UB@�A�Q�A�`�A��EAX�:@�G&Ak��A�mRA9��A���@}A��T?���B�A�T�A��A(��A�&aA���A�L�AE�A�U�Aq6BJeA<�@A�:@���Ai!,AɖA�Y�A�^A��B	 $A��k@��OA���A��zAAp�AU��A�z�A·�ALx"@�dX@��A`�YASa�AvA���A�!DC��+A���A���@�Ǉ@��<A�j@-@KX@�!�A͐�AЀ�AկfAY2@��_AkUA�}A9BsA�y%@~��A��'?���B�bAЉaAڄ�A(��A�>A�=�A��/AD��A߁�A��qB�A;	lA�m@��=Ah��Aɀ�A�|�A���A�k�B��A�j�@�GA�o�A�K�A?�ATPYA��oA�l�AK��@��@��A`�AS'�A��A�i�A�~8C�קA�/A���@��@��A��)@Y�@K��@��      	                  1   +            
   	      2   K                        )   k            4         D         +   	                  �         
         �         &         +      Z                  !      %   +                     %   -               !            1         #   -      !   /         '            '         %                                          !                           %                                       !                     !   %      !            %                                                               !N�'�N,�`N�(CN���N��O4�MNu�O�k�P��N���Nf�N#�Ny4NBN�NH�BOK��O��N1&�N�^Nș�N�yOa!EN��O#�N���O�1�O�yN]��O���P�OyeSO��~O{&�N�YN�GÕ�N��N rN��O�=�N��{N��>O7��N? �Nu�N�lO��Oj�O)��N#�N��O'|3N:_�O#gO��M��O���  �  �  8  �  �  M    �  4  �  M  �  �    �  `  .  �  �      s  1  �  �  
c  �  �  s  �    S  A  L  �  U  �    �  v  �  �  �  ~  �  �  �    �  \  �  �  �  '  L  <  u�����ě���j�D���o:�o�o;�`B%`  ;��
;D��;�o;�`B;ě�<o<�/=49X<t�<e`B<#�
<�o<�o<�t�<�t�=�w=y�#<�j<���<���=+<ě�<ě�=P�`<�/<�/=+<�h<�h<��=�E�=49X=,1=�j=49X=0 �=8Q�=D��=Y�>+=]/=]/=u=q��=��=� �=���=��!)---*)ot~�������~toooooooo20--15?BDBBCFB@52222^]\aabnvz{zxrnia^^^^^ZY`bnx{��{pnnb^^^^��� 
#,/542/(#
 �3+268BLDB63333333333����������

����uux���������������zu #00520/&# ����������������������������������������cgit}�����xtkgcccccc����������������������������������������rqrtw������������wtr )5BLONLKIB5)///<<HNLHA</////////���������

����|{|�����������~{��������������~~~~�����#+581)�����)-00)��������������������-./2<HMUVUPHC<5/----��������
�������'%),5N[gjkmme[NB;2*'"#/<BF</#"/4ITWUO;/"$eent��������������te )5BCGKNV[f[B5)+059?@BNTV^a`^[N5)�������������������� #'0<IMKIE@<70#������������������������#/<HPVHC=9/#�����������������������zy}���������zzzzzzzz�������	����������������������������������

���������������������������()6BDIKIEB6)���������������� "�������������������������#))&���������(+,)$���������

���������������������������������

��������~{uv{�������������~~��������������������;<@HJTUamnqomlaTH@;;)6BIMLJHFB6)5336>BDCB65555555555�z~�����������������ù������������������ùìêãìöùùùù��������������������������������������������)�6�<�B�G�B�6�)��������������	�����	���������۾�����𻅻��������������������}�x�x�x�����������m�y�����������������y�m�`�\�T�O�P�U�`�m�zÇÓÕÓÌÇ�z�y�s�z�z�z�z�z�z�z�z�z�z���(�4�A�M�Z�b�o�q�f�Z�M�A�(������s�������������������������x�h�Z�Z�]�f�s�-�:�F�I�S�U�T�S�F�:�/�-�!��!�'�-�-�-�-����*�3�*�*�����������������'�(�1�2�'���������������u�~ƁƉƁ�}�u�h�\�X�\�_�h�q�u�u�u�u�u�u�����������������������������������������h�t�w�t�i�h�^�[�O�K�O�P�[�]�h�h�h�h�h�h�Ľнݽ���������ݽнĽ������������������	��&�/�=�C�@�/�"��	�����������������z�����������������z�s�x�z�z�z�z�z�z�z�z���"�/�/�;�F�H�K�H�D�;�/�"�������f�s�����������������s�f�f�`�c�f�f�f�fčĚĦīįĮĦğĚďčăā�āăčččč�ݿ��������������ѿݿ��ݿ�ֿ�ƁƎƔƔƎƁ�u�l�u�}ƁƁƁƁƁƁƁƁƁƁ�4�A�M�W�Z�`�f�s�~�s�f�Z�M�A�4�(���(�4�����������������������������������뻑�����ûʻϻȻŻ»����������x�w�|�������`�m���������}�{�y�`�T�G�;�7�5�4�;�G�T�`ÇÓÔØØÓÇ�{�|�ÇÇÇÇÇÇÇÇÇÇ�H�T�^�a�b�b�a�T�H�;�/�"����������"�/�H�N�g�������������������g�Z�N�E�@�4�4�5�N¦¿������²¦�z�u�x�}��������$�I�V�[�Z�I�0�����������������5�>�B�M�I�A�0�)����������������5�Y�f�r���������������|�r�f�a�Y�S�M�Y�YÓØàãìùúóìà×ÓÈÇÁÇÐÒÊÓ���5�9�F�J�[�^�_�Z�N�5�(����������Z�_�f�s�|���������t�s�f�]�Z�M�I�M�X�Z�׾����� �������׾о׾׾׾׾׾׾׾��<�H�U�_�a�n�n�s�n�a�U�O�H�?�<�:�<�<�<�<ù��������������������ùìáÛØÛàìù���������ʾ;ʾž�������������������������'�4�@�K�M�U�M�@�4�'����������M�Y�f�r����������v�r�f�Y�M�@�=�:�>�L�M�"�.�;�?�=�;�.�"���"�"�"�"�"�"�"�"�"�"�ʾ׾�����׾ʾʾɾǾʾʾʾʾʾʾʾʽ������������������y�s�m�y�{�������������	��"�/�6�7�3�/�+�"���	������������	�#�0�<�I�U�b�i�n�x�n�b�U�I�<�0�#����#D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzD}D�D�D��/�/�#�����#�/�3�<�<�<�/�/�/�/�/�/�/��#�/�<�=�=�<�/�*�#�������������'�4�@�F�J�G�@�4�'���������������� �����������������b�n�{ŅŇœŔśŗŔŇ�{�s�n�a�U�Q�U�`�b�e�r�����������ͺκ����������~�r�X�R�Y�e�������������ٺ����������⼘�����ʼ�����ּʼ����������������� > V n >   R >  + e J C Z z ' : Z . H $ � Y h ! - * e L % P � = D \ Q L � T   ! T , 0 * , ' Q  S  / N J V W 4    �  [  �  �  �  }      u  �  �  R  �  �  �  �  �  `    �  �  }  *  �  �  i  9  �  �  :  	      "  �  �  1  `  �  5    �  �  K  �  �  Q  �  c  `  �  _  c  y  >    �  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  ?@  �  �  �  �  �  �  �  }  r  g  T  :       �  �  �  a  3    �  �  �  �  �  �  �  �  �  �  �  �  �  q  Y  ;    �  �  �  8  3  /  *  %      �  �  �  �  �  �  z  b  M  :  '      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  h  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  Z  4     �   �  �  �    -  B  K  L  G  >  9  )    �  �  �  D  �  i  �  M            �  �  �  �  �  �  �  z  \  >    �  �  �  �  �  �  F    �  �  �  �  �  {  [  2    �  �  4  �  �  9  �    -  3  3  3  /  *    
  �  �  �  Y    �  �  $  �  �   c  �  �  �  �  �  �  �  �  �  �  �  _  9    �  �  t  ?    �  M  J  G  D  A  ;  ,      �  �  �  �  �  h  E  !   �   �   �  �  �  �  �  �  �  �  �  �  �  w  d  O  :  $     �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  l  H  %    �  �  �        
  �  �  �  �  �  �  �  |  b  E  '  �  �  �  e  1  �  �  �  �  �  �  �  �  �  �  �  �  t  g  [  M  ?  2  $    �  .  w  �  �  .  Q  _  ^  U  C  (    �  �  T  �  ,  X  '    �    [  �  �  �    %  .    �  �  �  ^  �  X  \  �  $  �  �  �  �  �  {  w  r  n  i  e  _  Z  U  O  J  E  @  :  5  y  �  �  �  �  �  �  �  �  �  �  �  q  R  5         �  �            �  �  �  �  �    [  5    �  �  M    �  �  �            �  �  �  �  �  y  N    �  �  U    �  b  s  ]  I  B  ?  N  Y  \  ]  Z  U  O  I  B  5  %  �  �  P   �  1  1  0  0  /  .  )  %                �  �  �  �  �  �  �  �  �  �  �  �  �  }  g  P  9  "    �  �  v  0  �  �  a  >  \  ~  �    H  i  �  �  �  �  �  j  *  �  �  2  �  �  �  	B  	�  	�  
  
  
'  
O  
c  
Y  
J  
>  
<  
  	�  	/  �  �  �  �  N  �  �  �  �  �  �  �  k  K  )    �  �  �  �  V  &  �  9  �  �  �  �  �  �  �  �  �  w  [  >    �  �  @  �  �  x  0   �  T  p  l  H    �  �      �  �  �  �  �  Z    �  �  �   �  ,  �  �  �  �  �  �  �  �  �  t  M    �  �  >  �  3  j  �      �  �  �  �  �  �  n  Z  E  .       �  �  �  �  c    S  ;  "    �  �  �  �  �  �  �  �  t  T  1    �  �  ~  C  &  ]  �  �  �  �    1  @  2    �  �  |  5  �    r  �  �    )  4  @  H  K  J  C  9  -      �  �  �  y  I    �   �  �  �  �  �  t  \  B  '    �  �  �  �  j  ?    �  �  |  E  -  F  T  O  <  &      �  �  �  �  �  �  ~  O    �  �  B  �  �  �  �  �  �  �  �  �  �  �  �  i  D      �  �  �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  }  y  t  p  k  g  �  �  �  �  �  �  i  M  /    �  �  �  �  }  R  )    �  d  
�  �  �  Z  �    M  m  s  e  E  �  �  �  ;  Y  
O  	  b    �  �  �  �  �  �  �  �  �  |  c  F  &    �  �  n  7  �  �  �  �  W  '  �  �  �  R    �  �  �  {  H    �  }  0  �  �  �  �  �  7  �  �  �  �  �  �  �  M  �        �  9  	�  �  V  b  o  |  w  o  g  ^  T  J  >  0  "    �  �  �  �  �  c  �  �  �  �  �  �  �  �  �    w  n  e  [  Q  G  =  3  )    R  \  f  s  �  �  �  {  u  j  \  J  6    �  �  �  �  q  K  �  �  �  �  �  �  }  a  E  )    �  �  �  �  �  |  P     �    	  �  �  �  �  �  v  U  2    �  �  �  �  ~  r  d  T  B  B  4    �  d  �  J  �  �  �  u    b  u  Q  �  &    ;  �  \  a  a  Z  F  +    �  �  �  e  4    �  �  b  *  �  �  y  �  �  �  �  �  �  z  j  W  @    �  �  �  �  �  �  �  q  S  �  �  �  �  �  �  X    �  �    �    �  �  �  s  %  �  �  �  �  �  �  �  {  g  R  <  #  
  �  �  �  �  �  �  �  �  �  '    �  �  �  �  x  P  '  �  �  �  �  �  u  5  �  �  E  �  7  J  K  <    �  �  �  �  �  T    �  L  �  b  �  ^  �  �  <  6  /  )  #        �  �  �    !  7  M  {  �  �    V  u  p  b  E     �  �  p    �  F  �    
o  	�  �  �  �  �   u