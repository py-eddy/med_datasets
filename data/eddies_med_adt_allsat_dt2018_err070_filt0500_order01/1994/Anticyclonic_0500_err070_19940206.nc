CDF       
      obs    7   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?��G�z�      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P�*�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��j   max       =�-      �  d   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @D�33333     �   @   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��         max       @vr�G�{     �  (�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q�           p  1p   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�|�          �  1�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��C�   max       >O�      �  2�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�   max       B%�j      �  3�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�~l   max       B& �      �  4t   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >���   max       C�yU      �  5P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��O   max       C�w�      �  6,   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          i      �  7   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  7�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          A      �  8�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P�*�      �  9�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Z���ݘ   max       ?�0��(�      �  :x   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��j   max       =�-      �  ;T   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��
=p�   max       @DУ�
=q     �  <0   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����P    max       @vr�G�{     �  D�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @*         max       @Q�           p  M`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�q        max       @�|�          �  M�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         >�   max         >�      �  N�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z�G�{   max       ?�-w1��     0  O�                  )   _         &         C            !   1      ,                           	   #               i                                       
                  ,   NY�M��NO��P=�O@��P(O�1�N� kN*�qO��tOJ��O�qP�*�Ne@No�N#��PS��P8��O4�fO��XN��O<*�N��O���N�
Oz!#O�b�Pq��Or�O��#N��O/�N�`#OUJ�O�8�OD�N~3uO�CRN~�,N׍bOc�&O�,P-�N/ѻNY�M�l�Nc��N4�N:	Ne��N��N߉2O{��O�Y�NV1���j��t��ě���o;o;o;�o;��
;ě�;ě�<#�
<49X<49X<49X<e`B<e`B<u<�o<�o<��
<�1<�9X<�9X<�j<�j<���<���<���<���<���<���<�`B<��=C�=C�=C�=C�=�P=�P=�P=�w=#�
='�=0 �=@�=D��=L��=P�`=]/=]/=aG�=q��=u=�\)=�-opt������toooooooooo����

�������������@AFHUX`XUH@@@@@@@@@@��������������������')5BX[omlg`[PEB?;5/'	�#<Hahfbd^UH/	ssx���������������zs���������������������������������������������������������������������������������������������������������)5N[iffN5����^girtv�����tig^^^^^^��������������������������������������������#:BIB)����������������������������������������#<HPUansoa_OH</#����������������(&'()15:<BLNRVUN@5)(5*.6BCEB665555555555&'$)&*<HUr{��zn^=/&"##07<ED><0,$#"""""""/;><@BB?/"��������������������?;BNWi������[NJKQNI?�������������������������
/;?@<:/#
��			


						��������������������7BIO[]hhih[ODB777777��
#0;CA<60#
���������

����������������������������������������%5=BCBB;5)�������
���������������		������� $�����������������������qs����������������|q "?=@BJO[\[TOB????????>??BDOSQOB>>>>>>>>>>�����������ea`bhntvutpheeeeeeee��������������������
)31)#,/:/#��&)52,)�������������
�����������������������jfimzz��{zxmjjjjjjjj�L�Y�]�b�[�Y�W�L�D�E�L�L�L�L�L�L�L�L�L�L�_�f�`�_�T�S�I�F�F�F�S�U�_�_�_�_�_�_�_�_�	��"�(�*�"��	�� �	�	�	�	�	�	�	�	�	�	���������������	�����������������Z������$�$����������ü������������������a�m�z�������������������z�a�T�J�I�P�Z�a�O�[�h�q�y�|�v�[�B�:�)�"���
��� �6�OàâìííðìàßÓÇÆÅÇÒÓÜÜàà�Ŀǿ̿ǿĿ����������ĿĿĿĿĿĿĿĿĿ��;�E�A�H�\�_�T�H�>�"�	�����������	�"�/�;�������ùϹܹ��������ܹϹù�����������āčđĚěħĦĚęčā�t�h�g�]�h�q�tĀā��<�I�b�{ţūŦ�z�s�b�<�������ļ������u�{ƁƎƏƎƁƀ�u�h�e�f�h�k�u�u�u�u�u�u�*�6�?�B�6�*����&�*�*�*�*�*�*�*�*�*�*E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٿ��Ŀ�ڿÿ����y�`�T�;�0�!���"�.�G�T���5�B�[�h�t�h�N���������������5��������'�)�'������ݻܻӻܻ߻������� �"����	��������������������<�H�J�T�O�H�?�<�0�/�(�#��� �#�/�8�<�<���������$�2�0�$�����������������������������������������������������������A�N�g���������������������s�r�n�Z�A�3�A����������������r�p�n�r����������������"�/�;�H�T�[�a�e�i�e�a�T�H�;�/�"����	�"�ݽ���� � ����Ľ��������������н��������������������N�(��������5����������#�"�������������������`�m�y�����������y�m�`�V�T�I�@�=�?�G�T�`�	��"�$�"���	����������	�	�	�	�	�	����ܹ۹ع׹ܹ��������"�������׾޾����׾׾ʾȾ��þʾ;׾׾׾׾׾׼�������������ּͼƼƼʼͼּ��D�D�D�D�D�D�D�D�D�D�D�D�D�D�DzDgDdDnDuD��/�8�;�=�>�;�6�/�"���
�	���������	�"�/�����������������������������������������)�.�B�N�[�y�{�t�[�N�5�)�����)���������ʾ̾׾�׾ʾ�����������������������	��"�.�;�G�H�@�;�.�"�	���������_�l�x�������������x�l�_�F�-���#�-�M�_���ûлܻ�����������ܻԻлû����������M�Z�s������������m�M�A�(�����.�A�M�Y�e�r�u�s�r�o�e�Z�\�Y�T�Y�Y�Y�Y�Y�Y�Y�Y�����ɺκɺź���������������������������������� �����������������a�m�z�z���|�z�u�m�a�^�Y�a�a�a�a�a�a�a�a�Y�f�r�����w�r�f�_�Y�M�Y�Y�Y�Y�Y�Y�Y�Y�
��������
����������
�
�
�
�
�
���'�+�-�0�'��������������ݽ������ݽ׽ԽܽݽݽݽݽݽݽݽݽݽݾM�Z�c�f�j�h�f�e�Z�M�A�>�?�A�H�J�M�M�M�M��������������������������������������������������������ּʼ�������������Ź������������ŹŭŬŭųŹŹŹŹŹŹŹŹ G z 8 D } - 9 n M 8 @ = . F I G r T D A E H * @ 1 W f x N  � 1 c % ) : S ] u � r 3 9 R D O Q P v U b R 9 M @    J  O  g  4  B  �  E  �  ^  M  �  @  �  �  :  >  "  �  �  "    �    U  �    �  `  R  <  l  x  �  �  |  �  �  �  �  q  =  3  �  U  @    �  P  �  �  K      E  c��C���o�D��<u<T��=,1=���<�t�<#�
=49X=o<�`B=��w<u<�o<�1=D��=��=+=�o='�<�h<���=Y�<�h=0 �=49X=T��=C�=y�#<�`B=8Q�=�P=m�h>O�='�=�P=}�=,1='�=e`B=y�#=�o=D��=P�`=P�`=y�#=y�#=u=y�#=q��=�O�=���=�l�=��B��B#��BwgB�BZ$B��B��B"�B ��B@�B��B{CBB	�B��B[�B��B�B!��BʠB��Bn B׿Br�B%�jA�B!s�BW�B�BT4B3�B!�BǛB%�BB$�B3�B�BoHBkB0�BNZB&B+�B��B~B ��BТBa'B�BրBBx�B��A�8�B�(B#@�B��B�hB=�B��B��B"@�B ��BCNBóB��B��B	��B�4B@FB�dB�1B!�"B�B�UB��B�,B��B& �A�~lB!\�B	��B�TBA�BF*B ��B�B%BgB9�BA�B9�B�B?�B��B��BB�B��B��BC�BQmB ��B��B��B��B2�B)�B� B�A�aN?ЧK@�ubA�_gA���A�[cA�-A��A��Aw�A�o�>���A�b�A�B��A��TC�yUAlxTA���@��[A��;A��B@�A��3A��q@�1A��A-��A��,A�?SAj��AZ��?:j AR1A_sC��A��A���A���AL�vA]!b@�EI@��9A?�{?��<@+U@Qu�A���@ߴNA��?���A+VA=�&A� /A ��A�;�?՟}@���A�~�A��A�{�A��sA؁(A�i/AwI�A��J>��OA�|�A�{�B��A���C�w�Ai�A�jd@��}AҀ�A�B	�A��A��\@�o\A�z}A-��A�|�A���Ak�AW�r?O5�AR��A��C��]A�v�A�jWA��bAK?A^@�D@��:AB�t?��\@,
}@MqA��1@�A���?��)A*��A>�A�~�A �A�D�                  )   `         '         D            "   2      -                           	   $               i                                                         ,               ;      '   %         '         A            9   1                  '         '   ;                                             1                                 )                                          A            '   !                              ;                                             1                                 )   NY�M��NO��OD�0Oy�O��O ��NK��N*�qO[��N�ϜN���P�*�Ne@No�N#��O�!pO���O$JhO��N���O<*�N��N�7fNT)�Oz!#N�4HPq��N��sOZ�gN��O/�N�`#O	��O/�OD�N~3uOȉNZN׍bOc�&O�,P-�N/ѻNY�M�l�Nc��N4�N:	Ne��N��N�6O{��O�Y�NV1�  L  �  z  �  �      B    6  X  �  �  #  �    �  _  	  8  �  u    �  `  #  �  d  e  �  v  �  �  
  `       �  �  �  *    v  �  d  �  �  3  �    x  �  t  h  ��j��t��ě�;ě�;�o<49X=D��;�`B;ě�<��
<�o<u<49X<49X<e`B<e`B<�/=t�<�C�<��<�/<�9X<�9X=#�
<ě�<���=C�<���<�/=+<���<�`B<��=#�
=�\)=C�=C�=@�=��=�P=�w=#�
='�=0 �=@�=D��=L��=P�`=]/=]/=aG�=u=u=�\)=�-opt������toooooooooo����

�������������@AFHUX`XUH@@@@@@@@@@��������������������245BKNU[mjg[SIB@=852 #5<?T[^^XWUH/�����������������������������������������������������������������������������������������������������������������������������)5N[iffN5����^girtv�����tig^^^^^^���������������������������������������������#.7;8�����������
���������������������������" #&/<HNU[\USH<7/$#"��������������������(&'()15:<BLNRVUN@5)(5*.6BCEB665555555555;:9<=HU[aeda]UUH?<;;.&$+03<CC<<0........"/;><@BB?/"��������������������?;BNWi������[NJKQNI?�������������������������
)/6:;2/#
�			


						��������������������7BIO[]hhih[ODB777777
!#'099710#
��������

	���������������������������������������� )5:<<95.)����� �����������������		������� $�����������������������qs����������������|q "?=@BJO[\[TOB????????>??BDOSQOB>>>>>>>>>>�����������ea`bhntvutpheeeeeeee��������������������
)31)#,/:/# �	 )30+)    ���������
�����������������������jfimzz��{zxmjjjjjjjj�L�Y�]�b�[�Y�W�L�D�E�L�L�L�L�L�L�L�L�L�L�_�f�`�_�T�S�I�F�F�F�S�U�_�_�_�_�_�_�_�_�	��"�(�*�"��	�� �	�	�	�	�	�	�	�	�	�	��������������������������������������������������������������������
��a�m�z�����������������������m�a�^�U�Z�a�6�B�O�[�a�h�f�\�[�O�B�6�,�)�(�)�)�3�6�6àèìïìàÛÓÊÇÆÇÓ×àààààà�Ŀǿ̿ǿĿ����������ĿĿĿĿĿĿĿĿĿ���"�/�4�7�8�7�(�"��	�������������	�����ùϹܹ�����ܹϹù¹��������������h�tāčĐĚĝĚčā�t�o�h�b�h�h�h�h�h�h��<�I�b�{ţūŦ�z�s�b�<�������ļ������u�{ƁƎƏƎƁƀ�u�h�e�f�h�k�u�u�u�u�u�u�*�6�?�B�6�*����&�*�*�*�*�*�*�*�*�*�*E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�Eٿm�����������������m�`�T�G�;�(�.�;�N�T�m�5�B�N�g�l�n�f�[�N�5�)������������)�5�������&�(�'��������޻ܻڻܻ������������	����������������������/�<�H�L�I�H�<�;�/�#� � �#�(�/�/�/�/�/�/���������$�2�0�$�����������������������������������������������������������s���������������������{�s�g�g�`�g�o�s�s�r���������������r�r�p�r�r�r�r�r�r�r�r�"�/�;�H�T�[�a�e�i�e�a�T�H�;�/�"����	�"�������������ݽннƽнݽ������������������������N�(��������5����������"� �����	���������������`�m�y�������������y�m�`�T�Q�G�E�G�M�T�`�	��"�$�"���	����������	�	�	�	�	�	����ܹ۹ع׹ܹ��������"�������׾޾����׾׾ʾȾ��þʾ;׾׾׾׾׾׼��������	�������ּռ̼ͼּټ��D�D�D�D�D�D�D�D�D�D�D�D�D�D|D{DwD{D|D�D��/�8�;�=�>�;�6�/�"���
�	���������	�"�/�����������������������������������������)�5�=�B�N�[�d�g�p�p�g�[�N�B�4�)�%�"�&�)�����������ʾ˾ʾ���������������������������	��"�.�;�G�H�@�;�.�"�	���������_�l�x�������������x�l�_�F�-���#�-�M�_���ûлܻ�����������ܻԻлû����������M�Z�s������������m�M�A�(�����.�A�M�Y�e�r�u�s�r�o�e�Z�\�Y�T�Y�Y�Y�Y�Y�Y�Y�Y�����ɺκɺź���������������������������������� �����������������a�m�z�z���|�z�u�m�a�^�Y�a�a�a�a�a�a�a�a�Y�f�r�����w�r�f�_�Y�M�Y�Y�Y�Y�Y�Y�Y�Y�
��������
����������
�
�
�
�
�
���'�+�-�0�'��������������ݽ������ݽ׽ԽܽݽݽݽݽݽݽݽݽݽݾM�Z�a�f�i�g�f�c�Z�M�A�?�@�A�I�L�M�M�M�M��������������������������������������������������������ּʼ�������������Ź������������ŹŭŬŭųŹŹŹŹŹŹŹŹ G z 8 1 | '  W M ! 4 R . F I G r O B # I H * $ # W B x @ " � 1 c   : S P u � r 3 9 R D O Q P v U b L 9 M @    J  O  g  �  �  �  R  �  ^  �    �  �  �  :  >  j  �  ~  T  �  �      ^      `    �  l  x  �  '  p  �  �  m  |  q  =  3  �  U  @    �  P  �  �  K  �    E  c  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  >�  L  I  F  D  C  B  B  B  C  D  F  I  L  O  R  V  X  W  V  U  �  �  �  �  �  �  �  �  �  �  �  }  q  f  Z  O  D  8  -  !  z  |  ~  �  �  �  �  �  ~  y  t  o  j  c  Y  P  F  <  2  (  X  k  q  n  j  i  �  �  �  �  �  �  �  u  W  .    �  S   �  �  }    �  �  �  {  k  T  :    �  �  �  �  x  E  �  �  x  �  �  �  
                  �  �  �  P    �  )  �  i  	  	�  
!  
�  @  �  �        �  �  Z  
�  
"  	Z  0  �  �    �    B  <  2  %    �  �  �  p  ;    �  �  N    �  {                �  �  �  �  �  �  �  �  �  �  �  �  �  `  �  �  �    '  2  6  5  1  )    �  �  q  #  �  /  h  �  9  6  8  I  V  X  U  K  ?  ,    �  �  �  �  z  Y  0  �  L  3  Q  n  �  �  �  �  v  ]  1    �  i  n  9     �  �  E     �  �  �  u  {  o  X  R  .  �  �  �  �  z  n  T    �     L  #            �  �  �  �  �  �  t  c  R  @  .       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    	          �  �  �  �  �  �  �  �  {  d  N  2    �  �  �  \  `  z  �  �  �  ~  v  �  �  i  H    �  �  �  Y    �  /   �  x  �  �  �    )  I  \  ^  H    �  v    �  ^  �  @  ;  D  �    	      �  �  �  �  �  �  �  x  B  '    �  ~  �  J    u  �    .  7  .      �  �  �  S    �  ,  �    >  ^  �  �  �  �  �  �  �  �  �  �  �  [  0    �  �  Q  �      u  i  \  O  @  1  %        �  �  �  �  �  �    i  S  =    	    �  �  �  �  �  �  �  �  �  �  |  h  O  5       �     =  M  `  �  �  �  �  �  �  �  �  �  �  �  d    �  �  �  P  U  Z  _  ]  Y  V  M  B  7  +        �  �  �  �  �    #      �  �  �  �  �  �  �  �  �  z  ]  =      �  �  �  _  h  t  �  �  �  �  �  �  �  �  �  �  �  x  b  J  ,      d  I  *    �  �  �  T    �  �  �  d  h  C    �  �  �  K  ^  a  d  a  [  T  M  F  ;  0  !    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  j  )  �  �  H  �  �  �      v  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  `  G  .                �  �  �  �  g  7  �  �  �  �  �  �  �  z  g  S  ?  )    �  �  �  �  b     �  �  �  �    
  
      �  �  �  �  �  u  .  �  �  N    �  �  �  %  �  �  +  U  ^  A    �  �    y  �  �  
�  	�     �            �  �  �  �  �  �  �  n  X  B  '     �   �   �   d                  �  �  �  �  �  �  �  x  a  J  3    �  �  �  �  �  �  �  �  �  �  �  �  d  '  �  }    �  �  +  �  �  �  �  �  �  �  �  �  �  �  �    j  U  @  ,       �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  *  '  !         �  �  �  �  �  �  �  V    �  ,  �  9   �            �  �  �  �  �  q  K    �  �  '  �    f   �  v  K  *    �  �  �  �  �  �  �  �  �  r  i  R  +  �  �    �  �  �  �  �  �  �  �  �  �  �    .  :  G  S  a  o  |  �  d  ^  X  R  L  F  @  :  4  .  !    �  �  �  �  f  +   �   �  �  �  �  �  �  �  �  }  s  h  ^  S  H  6    �  �  �  �  l  �  �  �  h  B    �  �  �  �  q  D    �  �  �  h  9    �  3       �  �  �  �  p  =  �  �  t  (  �  �  +  �  r     �  �  �  �  �  x  j  ]  O  A  3    �  �  �  D    �  �  i  8        �  �  �  �  �  �  �  �  �  �  |  k  W  D  8  /  '  x  w  w  v  u  t  s  q  o  m  ^  A  %    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  q  M  %  �  �  �  �  Y  1    �  t  R  (  �  �  �  K    �  �  �  u  P  *    �  A  �    ?  h  F    �  �  �  z  I  ]  =    �  �  >  �  �  .  �  0  w    �  �  �  �  �  �  �  �  r  _  L  9  &            