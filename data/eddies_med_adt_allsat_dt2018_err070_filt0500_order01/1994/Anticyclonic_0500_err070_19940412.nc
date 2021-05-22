CDF       
      obs    ;   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�            �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�Zo   max       P��      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       =�E�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?+��Q�   max       @E��
=p�     	8   p   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @vqp��
>     	8  )�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @1         max       @Q�           x  2�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�K           �  3X   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��o   max       >8Q�      �  4D   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�s   max       B+��      �  50   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��Z   max       B+�1      �  6   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?��   max       C�o      �  7   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?��   max       C�hq      �  7�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          ~      �  8�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          E      �  9�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          C      �  :�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�Zo   max       P��B      �  ;�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?���Y��}   max       ?ۤ?��      �  <�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       =�      �  =|   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?.z�G�   max       @E��
=p�     	8  >h   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�
=p��    max       @vqp��
>     	8  G�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q�           x  P�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�T           �  QP   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         ?,   max         ?,      �  R<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?x*�0��   max       ?ۢ�w�kQ     p  S(            <                            G      0      b            K            ?                  7                              T               }                           u            NPKNi��N�lP��NT�No�O�+O]Q4PNF N�IMɬ�Pk7TN㒌P���O��P��O���O:�WN��PkгO�b�N���OpP@�:O��zNd�sN|��N�P5s P^;NN�9O�g�M�ZoOC�N�U�NOy�O۾�N�e�O�R P>��N�N�.%NIU�N��BP�qN9y�N}AOZv�N(i�N)��Om�Oe�}N���O��JN�iO:�NW#�N�I9��j�D���D���o�o��o%   ;o;D��;D��;�o;�o;��
;��
;ě�;�`B<t�<#�
<#�
<49X<T��<e`B<e`B<�C�<�C�<�t�<�t�<���<���<��
<��
<�1<�9X<�9X<�9X<�9X<ě�<���<���<�h<�<�<��<��=o=+=+=C�=\)=�P=��=#�
=0 �=P�`=�%=�C�=�hs=�t�=�E���������������������gcbgitx���utggggggg�������������������������5Nx���tYN5������������������������6<9<EHOMH<6666666666����������������������
")*/47/%"
��������/99HahaPU<#����^afhtu}zth^^^^^^^^^^).595/)/,%0<??<00//////////��������� ���������������������������������6[aO6�������&).6Baf[ZSNGB6 ���)NUfdWT5��������tng`YV[gt��������������������������A<BO[`[ZOBAAAAAAAAAAFDLZey�����������gTF�������� �����*-67;BOWZOMB=6******������" ������&/5E[nwyvdN5)&(#&,8ABNVY[acdZWB5)(~��������������	���������������������������|}�����������������|����������������������������������...--;Ham}zwmiaTH;7.55ABNPPNB55555555555������������������������������������������������������������
#/<HUXU</#
���aanz�����zzynlhaaaaa������������������������������������������������������������UOH></# #+/<HTUUUUrmot�����trrrrrrrrrr�������������������������)6GVYUIB)�%!)6BBEB6)%%%%%%%%%%HHJTaaehaTMHHHHHHHHH�w|�������������������������������������aaflnsxz�znaaaaaaaaa������#*-+#
����/1<HXbd^`d]UI><?<60/����������������������������

�����`aa[Zajmruz{{ztmjba`jmqoz������������zmj"#/5<DH</+(#""""""""&#!$)36BCB@9866)&&&&�=�I�J�V�Z�V�I�=�0�+�0�4�=�=�=�=�=�=�=�=��������������������������������������������������������������������������<�I�nŀŘŇ�n�U�<�0�#��������ĺĭĳ�����������������������������������������������������������������������������������G�T�`�m�s�z�m�j�l�`�T�G�;�3�/�(�&�-�;�G�������������y�m�c�`�T�P�G�G�M�Q�]�`�m���"�T�a�r�q�l�a�H�2�&����������������	�"�L�Y�e�l�f�e�Y�L�F�@�L�L�L�L�L�L�L�L�L�L�Z�f�i�h�i�f�_�Z�Q�M�I�L�M�V�Z�Z�Z�Z�Z�Z�r�����������}�r�r�r�r�r�r�r�r�r�r�r�r�N�g�[�B������������������)�5�N���������	����������������������������"�S�Y�S�Z�W�L�2��������������������	�"���̾׾�������׾;ʾ����������������	��;�Y�m�z�w�\�T�/������������������	ùöôñãÞàçù��������������������ùăčďČđčďčā�t�S�O�G�[�\�h�t�wĀăàìðòòìààÜÙàààààààààà�A�N�s�����������������s�N�5�(�#��+�5�A�ѿݿ�������� �������̿ƿÿʿѻлܻ�����������߻ܻѻлλллллл�ƧƳ������������������ƳƚƗƎƁ�uƁƎƧ�G�T�`�j�������������y�T�G�.���
��.�G�������$�=�M�[�I�=�0�$��������������򿸿ĿпϿĿ��������������������������������
��
�������������������������������Óàâì÷ùþùöìàÓÇÆÆÇÌÊÓÓ�4�A�Z�����������������f�R�A�(�*��%�4�������л�����ܻû������x�l�h�k�x���f�f�i�k�f�_�Z�M�G�L�M�U�Z�f�f�f�f�f�f�f�/�;�H�T�a�j�j�e�c�a�_�T�F�;�0�&���"�/ƁƂƎƎƐƎƁ��zƁƁƁƁƁƁƁƁƁƁƁ��������������������������������������ҿ"�.�;�G�L�G�E�<�;�.�$�"���"�"�"�"�"�"���
������
���������������������������(�8�=�8�"����������ӿ˿ѿ�����A�C�H�K�F�A�5�(�&����"�(�5�@�A�A�A�A�	��"�;�G�T�`�T�G�;�.��	���������	�ûܻ���@�P�f������M�4���ܻ��������/�<�H�T�S�H�@�<�/�(�&�+�/�/�/�/�/�/�/�/ED�D�D�D�D�D�D�D�EE
EEEEEEEEEƚƧƳƻƽƳƧƚƙƙƚƚƚƚƚƚƚƚƚƚ�����ʼ˼ʼƼ��������������������������������������ռ����ּ����������~�{�|�����������������������������ݿ޿�����ݿѿǿȿѿݿݿݿݿݿݿݿݿݽݽ���� ���ܽĽ��������������Ľн�ÓÝÓÎÇ�z�n�a�[�a�n�zÇÈÓÓÓÓÓÓE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eپ���������þ���������������s�h�c�f�s���������������ùìäìù�������������y�������������������}�y�u�u�w�v�y�y�y�yD�D�D�D�D�D�D�D�D�D�D�D�D�D�DDuDtDxD�D�������������������������ŹŴŭŪŭŹ�������������������������������������ǔǕǘǔǐǈ�{�o�t�{ǈǓǔǔǔǔǔǔǔǔ�~�������������������������~�~�|�~�~�~�~ A * _ 0 U G 8 E 9 3 P P W 2 [ 9 : * C [ B _ k L , Z C g * > . p < d T ( \ g < L m 6 I " 5 5 > C � � K ( \ � ' v > c ]    h  �  V    g  :  >  �    _  �         {  �  O  #  �  E    �  �    Q  �  p  �  �  0  �  �  /  %  �  �  �  �  �  n  �  �  )  ^  �  i  Z  z  �  �  W  �  �    m  �  �  �  ˼�o;o%   =ix�:�o:�o<D��<�t�=\)<u<t�;ě�=���<�o=]/<�j=��<�h<�9X<u=�-=,1<���=+=��
<��<�9X<ě�=\)=0 �=���<�j=#�
<ě�=T��<�`B<�`B=P�`<��=<j=�x�=0 �=P�`=��=#�
>!��=��=�w=T��=,1=49X=�%=�hs=m�h>8Q�=��P=�E�=��
=���B0�B	��B�IBr B�LB�BZ�B��B�B��Bl�B%ڭB�@B��BE�B�B�5B
'�B�QB�`B
e�B��B8�B'�B��B�BN�B9�B!�TB��By�B�A�sBǠB�B�B�BM_B�B!C(B!"�B��B�|B
YB"�B��B��A�ӷBW�B�vB�,B$6gB(�B+��B��A�D�B �B[�B�JB@B	�%B��Be�B��B=oB�jB��B�aB>nB�EB%��B�HB��B�TB�mB��B
L~B4BB�YB
�
BH�B?�B?�B@SB��B@mB;�B"8�Bt�BAFBh�A��ZB�B;`B:�BĩBS�B>�B ԼB!@+B�eB��B
: B!��B�B�{A��qB�BB�B�&B#�DB7B+�1B��A�|�B>rB��BBB
�qA��,A�J�A���As5A�4Ad��Al'�A�υ?��A?Fw@�*�A�aBA�u8A�O�AP.^A��FA���A�ĩA�-1A��A��@��B��Ah?�B	QAw��A���A�_�A?�&@�؍A?G	A�y�B]jA�D�AaģA�T�A���A��A_%2@ģ�AÈ$C�RnB��@�+'@��A�A|�sA%{�AȆYC�oAHpA�)�A�]C���A�,A�[�B��@3tB
̊A�� A���A�o�AsP�A�X�AcKDAl��A��q?��A? @���A�k�A�ւA��AO�A���Aϫ�A�{^A̝A�WA�t@���B��Ag�B�CAw	�A���A�hkADf@�?#A>��A���BB�A�v3Aa!�A�<A��=A��A]p@�	�A�vC�XB6�@��U@���A�g�A|��A'#�A�Y�C�hqAI��A�zA�C��A�n<Aњ�BD�@�y            <               !            G      1      b            K            @                  8                              U            	   ~                           v         	               E               5            7      A      5            3   !         /   !            1   '                     '         7               %         !                                          C               1                  /                  #            %   !            +   %                     '                                 !                              NPKNi��N�lP��BNT�No�OU�O	qPKq�NF N�IMɬ�OYG�N���P<V�OSO��O���N���N��O���O�1Nt#�O�rO�J O��zNd�sN|��N�P'��O�:NN�9Oo
PM�ZoN�lcN�U�NOy�O۾�N�e�O�R O�T�N�N�.%NIU�N��BO���N9y�N}AOZv�N(i�N)��Om�N��aN���O	WN�iO�iNW#�N�I9  �     =  �  �    }  x  6  �  h  �    y  B  �  �  M  �  4  ^  �  �  �  �  }  �     �  Z  `  �  �  �  I  :  c  �  .  �  	5  �  �  �  "  A  ?  �  8  �  �  P  e  �  �  �  e  {  ż�j�D���D��%   �o��o;o;ě�;�`B;D��;�o;�o=D��;ě�<�1<T��=�+<#�
<u<49X=�P<�C�<u<�1<��<�t�<�t�<���<���<�1<�h<�1<�j<�9X=\)<�9X<ě�<���<���<�h=��<�<��<��=o=���=+=C�=\)=�P=��=#�
=]/=P�`=�=�C�=��=�t�=�E���������������������gcbgitx���utggggggg������������������������)5Nt��kWN5������������������������6<9<EHOMH<6666666666������������������������ 
$"##
 ��������.4.AU`^<#���^afhtu}zth^^^^^^^^^^).595/)/,%0<??<00//////////����������������������������� ��������������<KIB6������)%&)*56BHOTSOMHB@62)		)5BFEA;5)�����tng`YV[gt��������������������������A<BO[`[ZOBAAAAAAAAAATQTYa�����������vg[T�������
����1068<BOVWOLB86111111���������
&55Bamnje[NB5)(#&,8ABNVY[acdZWB5)(~��������������	�������������������������������������������������������������������������������./;HTamrtmhaTH;90/0.55ABNPPNB55555555555������������������������������������������������������������
#/<HUXU</#
���aanz�����zzynlhaaaaa������������������������������������������������������������UOH></# #+/<HTUUUUrmot�����trrrrrrrrrr������������������������)6@FGG?6)�%!)6BBEB6)%%%%%%%%%%HHJTaaehaTMHHHHHHHHH�w|�������������������������������������aaflnsxz�znaaaaaaaaa������#*-+#
����648<HLUYXUH<66666666��������������������������� 	

	 ������`aa[Zajmruz{{ztmjba`yuz�������������{zyy"#/5<DH</+(#""""""""&#!$)36BCB@9866)&&&&�=�I�J�V�Z�V�I�=�0�+�0�4�=�=�=�=�=�=�=�=��������������������������������������������������������������������������<�d�sńœŕŃ�n�U�<���������ĽĴ�������������������������������������������������������������������������������������;�G�T�^�`�d�e�f�`�T�G�;�9�2�.�+�(�-�1�;�m�y���������������y�m�f�`�T�Q�T�W�`�d�m�	�"�;�T�`�m�l�a�T�H�/�&��	�����������	�L�Y�e�l�f�e�Y�L�F�@�L�L�L�L�L�L�L�L�L�L�Z�f�i�h�i�f�_�Z�Q�M�I�L�M�V�Z�Z�Z�Z�Z�Z�r�����������}�r�r�r�r�r�r�r�r�r�r�r�r��)�5�B�I�N�C�5�)�����������������������������������������������������"�/�@�J�N�L�B�/��������������������	�"�����ʾվ׾����ݾ׾ʾ������������������	��"�/�;�A�E�A�/�"��	��������������ùöôñãÞàçù��������������������ù�h�tāąĈĆăā�z�t�h�`�g�g�h�h�h�h�h�hàìðòòìààÜÙàààààààààà�N�Z�g�s�������������g�N�A�8�/�-�5�<�G�N�ѿ�����������������Ͽʿǿѻлܻ������������ܻһлϻллллл�ƚƧƳ������������������ƿƳƧƦƚƗƓƚ�G�T�`�m�y���������y�T�G�;�0�'�!�"�'�/�G�������$�=�M�[�I�=�0�$��������������򿸿ĿпϿĿ��������������������������������
��
�������������������������������Óàâì÷ùþùöìàÓÇÆÆÇÌÊÓÓ�4�A�Z�f�����������������f�U�F�5��(�4�������лڻ����ܻлû������x�v�z�����f�f�i�k�f�_�Z�M�G�L�M�U�Z�f�f�f�f�f�f�f�T�a�h�h�f�c�b�a�T�H�;�1�'� � �"�/�;�H�TƁƂƎƎƐƎƁ��zƁƁƁƁƁƁƁƁƁƁƁ���������������������������������������ҿ"�.�;�G�L�G�E�<�;�.�$�"���"�"�"�"�"�"���
������
���������������������������(�8�=�8�"����������ӿ˿ѿ�����A�C�H�K�F�A�5�(�&����"�(�5�@�A�A�A�A�	��"�;�G�T�`�T�G�;�.��	���������	����'�4�@�F�M�M�@�4�'��������޻���/�<�H�T�S�H�@�<�/�(�&�+�/�/�/�/�/�/�/�/ED�D�D�D�D�D�D�D�EE
EEEEEEEEEƚƧƳƻƽƳƧƚƙƙƚƚƚƚƚƚƚƚƚƚ�����ʼ˼ʼƼ������������������������������������ļм׼ڼ׼ʼ��������������������������������������������ݿ޿�����ݿѿǿȿѿݿݿݿݿݿݿݿݿݽݽ���� ���ܽĽ��������������Ľн�ÓÝÓÎÇ�z�n�a�[�a�n�zÇÈÓÓÓÓÓÓE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eپ���������þ���������������s�h�c�f�s�������������������������������������y�������������������}�y�u�u�w�v�y�y�y�yD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�������������������������ŹŴŭŪŭŹ�������������������������������������ǔǕǘǔǐǈ�{�o�t�{ǈǓǔǔǔǔǔǔǔǔ�~�������������������������~�~�|�~�~�~�~ A * _ + U G > / : 3 P P U 0 ` + * * @ [ * _ ^ 6 # Z C g * ? + p 3 d ? ( \ g < L @ 6 I " 5 , > C � � K ( & �  v . c ]    h  �  V  �  g  :  �  .  Z  _  �    �  �  �    �  #  �  E  @  u  �  U  *  �  p  �  �      �  �  %  �  �  �  �  �  n    �  )  ^  �  2  Z  z  �  �  W  �  �      �  Q  �  �  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  ?,  �  �    j  T  >  %  
  �  �  �  �  T  '  �  �  �  k  >               �  �  �  �  �  �  �  �  �  t  d  _  ^  \  [  =  /  !      �  �  �  �  �  �  �  �  �  y  j  Z  J  ;  +  �  �  �  g  q  �  �  �  �  z  b  F  &    �  �  s    �  C  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  I  )  
        �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  Z  h  s  |  w  o  b  Q  >  '    �  �  �  �  �  �  �  �  �  !  4  P  g  t  v  m  ]  I  0    �  �  �  o  .  �  �  �  �  �  (  6  5  0  *  "      �  �  �  �  �  �  �  J  �  �   �  �  �  �  �  �  �  �  �  �  �  �  %    �  �  �  �  p  J  $  h  c  ^  Y  P  7      �  �  �  d  ;     �   �   �   }   \   ;  �  �  �  �  �  �  {  r  h  _  T  G  :  -          �   �   �  p  �  C  �  �  �        �  �      �  �  W  �  �  :  �  c  p  w  o  e  X  H  8  &    �  �  �  �  �  f  ;    �  �  t  �  �    *  =  B  5      �  �  �  {  C    �  d  �   �  �  �  �  �  �  �  �  �  �  �  �  �  w  R  *    �  �  }   �  �  �  /  �  �  6  c  �  �  �  �  �  �  �  #  �  �  �  i    M  L  @  3  )    �  �  �  �  �  t  X  9  
  �  �  ~  t     _  `  f  t  |  �  �  �  �  �  l  J  (    �  �    �  �  �  4  :  @  E  K  Q  W  ^  d  k  q  v  {  �  �  �  �  �  �  �  �  �    &  I  Z  ]  Z  P  =    �  �  7  �  ;  �  �    U  x  �  �  �  w  f  T  A  0    �  �  �  r  &  �  Y  �  l  \  �  �  �  �  �  �  �  �  �  d  <    �  �  �  j  1  �  �    C  m  �  �  �  �  �  �  �  l  J  #  �  �  �  Y  *  �  �  �  �  �  �  �  �  �  �  �  �  �  i  0  �  �  7  �  d  �  �  ]  }  z  t  j  ]  L  =  @  5  "      �  �  �  �  �  �  �  i  �  {  s  k  c  Z  Q  G  >  4  &       �   �   �   �   �   h   J     �  �  �  �  �  �  �  �    g  O  >  2  &        �  �  �  �  }  Y  3  
  �  �  �  O    �  �  �  �  �  �  u  D    O  Y  O  @  ,    �  �  �  �  �  �  �  �  �  �  u  B    �  J  W  ^  `  [  H  3    �  �  �  t  /  �  �  N  �  �  �  �  �  �  �  �  �  �  �  �  �  �    ~  }  {  z  y  w  v  t  s  �  �  �  �  �  �  �  �  �  �  i  D    �  �  3  �  �  0   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  r  h  �  �  �  �    0  B  E  H  E  9     �  �  �  �  5  �  P  �  :  3  +  $          �  �  �  �  �  �  �  }  i  T  ?  +  c  [  S  J  B  9  ,        �  �  �  �  �  g  A    �  �  �  �  �  �    j  R  3    �  �  �  _  Y  &  �  �  �  =  �  .  ,  )  '  $          �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  S  /    �  �  .   �  Y  �  ?  �  �    �  	  	5  	  �  y  �  v    �  :  ?  �  �  �  �  �  �  �  �  �  �  �  �  k  A    �  �  T    �  C  �  �  |  ]  9    �  �  �  M    �  �  D  �  �  *  �  f    �  �  �  �  �  �  �  �  �  w  `  I  0    �  �  �  u  S  1    "            �  �  �  �  �  �  �  �  n  C    �  �  �  �    �  �    .  @  <    �  B  �  �  &  )  
  �  �  �  �  ?  2  &      �  �  �  �  �  �  �  �  }  n  _  U  M  D  <  �  �  �  v  h  [  N  A  4  &      �  �  �  �  �  �  �  e  8  !            �  �  �  �  �  �  o  I  $  �  �  z  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  r  ^  I  �  �  �  �  l  Q  6    �  �  �  �    \  9     �   �   �   �  P  O  H  >  0  L  @  2      �  �  �  T  1    �  �  �  t  �  �  l  ?  [    U  a  b  X  A  '    �  �  �  L  �  :  `  �  �  �  �  �  �  �  �  �  �  �  �  j  L  %  �  �  �  l  G  �  "  u  �  $  p  �  �  �  �  �  [  �  /  F  L  /  
k  �  p  �  r  R  2      �  �  �  �  �  y  i  Q  '  �  �  �  n  =  2  !  ^  ]  Q  >  '    �  �  �  k  -  �  �  k  <      �  {  j  Z  J  :  *      �  �  �  �  �  �  �  �  z  _  =    �  �  �  �  r  [  H  4    �  �  L  �  �  N    �  g     �