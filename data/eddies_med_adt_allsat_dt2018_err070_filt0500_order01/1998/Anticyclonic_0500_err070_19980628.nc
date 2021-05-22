CDF       
      obs    9   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?ļj~��#      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M�U   max       P�2�      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �ě�   max       =��      �  t   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @E�
=p��     �   X   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��\(�    max       @vk�
=p�     �  )@   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0�        max       @Q�           t  2(   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @��          �  2�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ���
   max       >_;d      �  3�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��l   max       B.�      �  4d   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�|=   max       B/;       �  5H   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�}   max       C�"�      �  6,   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��J   max       C�H      �  7   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  7�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  8�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3      �  9�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M�U   max       Po�      �  :�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��Xy=ـ   max       ?�ڹ�Y�      �  ;�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �D��   max       >o      �  <h   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�Q��   max       @E�
=p��     �  =L   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @vk�
=p�     �  F4   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q�           t  O   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�_        max       @��          �  O�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         E.   max         E.      �  Pt   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�z�G�{   max       ?�7KƧ�     �  QX   "   �                           	                  	         1               C             U      3      r            8      *   
      F         
   +   �            #         ^      9   O�wP�2�O�OE�N�%�OK�N'wKNd�N��N�N#�!O���O��O�r�NN��N�O'�}N�/VO�B4Pc#N9N�O�/O$�	N��Pd}�N��N@�O�9�P��N��P W<M�UP W�N�M�N�<�O-g@OΔ�N�\~O���NABN��PR�NgpkOh��N��P�O�\�Nc"�NJ��Op`�O��NXP7N�C�PQ�M�+Ok �N	[�ě���t���C���o;D��;D��;�o;�o;��
;ě�<t�<t�<#�
<T��<T��<�o<�t�<�t�<���<���<��
<�1<�1<�9X<�j<ě�<ě�<�`B<�`B<�`B<�`B<�<�=o=t�=t�=��=�w=�w=#�
=#�
=0 �=H�9=L��=e`B=e`B=ix�=ix�=m�h=�O�=�hs=�t�=���=���=�{=�;d=���������	
�������������
%Ha{}naU</
���������������������g`_cgjtv��������tmggjkt���������vtjjjjjj���)6B622)&�vv��������vvvvvvvvvv����������������������������������������������������������������������������������
#,.0-'
�������	"/:BEILG;/"	��&%'+<HUailqnona]U</&z~����������zzzzzzzz36BBO[a^[OB633333333���������������������������������������������������������������
/6LTWUI<#
���()0<IJIE<0((((((((((������
!
�����!#%)+/8<HUab_]YUH</!���������������������)4?E?5���##01<BA<30##040-#�  ��������������(&)5Ng~������t[N<5+("#/;<FE@</#""""""
3BO`lpkg[B6)��������������������������������������������������������SRW[gtu|v|tpg[SSSSSS������������������������������������������������������������)5?KRTSWNB5�����������������������opx{���������{oooooo����)5?AFEB<2��#)"������������������������������������%*-&��������������

������������������������tst���������|ttttttt������)*11)���gglt����������{tpkggpqsz�����zpppppppppp������	

���������	6BTYWQB6)�vrzz������|zvvvvvvvv�����������������������������������˻����ûѻԻλû������t�l�a�^�e�l�m�x������(�N�s���������p�F�(��ӿ��������ݿ��ĳĶĳİĳĸĸĳĦĚčć�v�wāčĚĦĬĳÇÓàìøù��ùðìàÓÇÂ�z�t�r�zÆÇ�g�t�x��{�t�h�g�e�[�X�T�[�^�g�g�g�g�g�g�A�N�Z�l�p�s�����|�p�g�`�Z�N�C�A�6�0�8�A�������������������������������������������������ɾ������������������������������N�U�Z�a�e�a�Z�T�N�A�?�7�A�G�N�N�N�N�N�N�������������׾־׾��������B�O�[�\�[�X�O�B�7�A�B�B�B�B�B�B�B�B�B�B�A�M�Z�f�q�o�V�A�4�����������4�A�T�a�m�r�v�u�m�a�T�H�;�/� ���#�/�5�;�T�s����������Ǿ�������s�f�_�a�f�j�o�p�s�<�H�P�N�H�?�<�/�*�(�/�7�<�<�<�<�<�<�<�<�z�~�����������z�x�t�s�x�z�z�z�z�z�z�z�z�s�����������������s�f�b�b�`�a�f�i�s�s�s�������s�n�f�Z�T�Z�^�f�j�s�s�s�s�s�s����������������ŹŭŠŔŎŋōŘŠŹ���ҿG�T�`�m�y������m�G�.������ �.�;�G���������������v�v�����������T�a�z�������������z�m�a�W�H�E�=�6�>�H�T�����������������������������������������T�`�c�e�m�y�~�y�m�`�T�L�G�?�G�H�T�T�T�T����<�U�b�{ŉőœň�{�U�#�
����������������������������ߺֺҺ˺ֺغ����ּ�����ڼּҼռּּּּּּּּּּּʼ����������}�������������ʼҼּۼ���"�a�m�u�n�\�;�.����������������������"��������	��������������������������޾����ʾ׾����	���	��׾���������������*�6�C�K�C�6�*������������ùܹ�	������Ϲù����x�l�t�������ÿT�`�m�s�y���������y�q�m�j�`�V�U�T�R�T�T�m�y�����������y�m�`�W�`�e�b�m�m�m�m�m�m�;�H�T�W�a�m�y�s�m�b�T�H�;�/�*�(�)�/�2�;�O�h�tčěĬĵĺļĸĳĦĚč�t�O�B�7�B�O���)�*�1�)�'������������������B�O�X�m�{�{�t�h�[�B�)��	���� �)�5�BàìùúùôìàÓËÓ×àààààààà�������!�'�!�������������������������������0�F�=�&�����ƳƦƚƎ�n�kƧ���������������������������������������������/�6�<�?�;�3�/�#��
���������������������������������������s�g�d�g�o�������Z�`�W�Z�T�R�K�A�5�����������(�N�ZD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|D}D�D�D��y�{�����������y�p�p�r�x�y�y�y�y�y�y�y�y�F�S�V�X�S�J�F�>�:�7�9�/�:�D�F�F�F�F�F�F�)�5�B�N�\�g�t�g�Y�M�F�A�5�)�)ǔǗǡǬǪǡǜǔǈ�{�z�o�k�l�o�v�{ǈǓǔ�M�Y�f�r�f�`�Y�M�I�B�M�M�M�M�M�M�M�M�M�M�S�_�l�x��������x�l�_�S�O�L�S�S�S�S�S�S�л����)�-�(�'����ܻŻ������������н��!�"�.�7�.�!������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E{EvE�E�E*E7ECEPEREPEPECE7E6E*E*E*E*E*E*E*E*E*E* L B = ; < < C S A M H K ! 4 I � F k c S G 4 A \ ; 3 u * J . " � 2 6 < + E ^ 2 > h T A 6 U R % a ^ n < D V B y $ `    >  ]  /  R  �  �  N  v  �  E  7  �  d  x  p  �    �  7    l  (  �  �  �  �  w  '  �  �  �  �  �  �  �  s    �  Z  W  �  �  �  �  8  )  �  �  �  `  ,  l  �  �  ]  �  O<t�=�S����
<�C�<#�
<�j<#�
<o<49X<o<�C�=��=��=\)<���<��
<�/<�9X=t�=�7L<���='�<�`B<���=�Q�=C�<�/=u=�`B=#�
=��w=\)>n�=�P=@�=T��=�v�=0 �=��
=L��=Y�=�`B=]/=�\)=��=���>_;d=�o=}�=�^5=��=��w=�Q�>-V=�Q�>)��>oB#{B rB�NB	�FB
I	B�<BޓB#t�B��B?qB�xB$_�A��lB�BD�By�B��B��B��B�B&sBB�B`.B��B%�aB%
UB"�|B�<BF�BV�B�tB٤BT�B	1#B�B��B��B��B!�}B)�BAgB.�B��B�B�4B�B-)B�Bi>B
@Bd�B}B2qB[BnoB^B#(1B?|B�lB	��B
O�BFB�B#w"B�sB@&BZ�B$��A�|=B�@BAWB\�B��B��B�7BS�B&:�B�IB>�B��B� B%W�B%M�B"�)B	�BE^BfB��B�_B>�B	@�B��B��B��B�bB"?�B(�=B�B/; BÆB �PB*B��B-<fB��B �B
@?BA�BAB?�B@�B�BHU@���A���A�i�A���A��A�\l@��AK�,A���AV/�A�2yA7>OA��<AH�A�J>A���AD(XAAҞA��NAczj@�g0A��A�jMAh�CA��@E�dA�Y@���A��A��AP6A�x>�}Aj��Al��A�b�Aݴ�AԒ~A�h<A��@_`]B�;Arq�A�9A�q�A�'0C��FA(@�?�A�RB�A@ٗ�@��S@���A%�C�"�C���@�`OA��Aߞ�A��A���A��o@�Y�AJ�ZA�w�AV�,A��A6�A�s�AI�A�a�A�k�ACWAB�iA�tAb��@�A�WiA�q5Ah1�A냔@EB�A^�@���A�9oAь�AN��A�P�>��JAk�_Al��A�ՠA݋yA�^�A�wGA�t�@[��B�~AsL�A�}�A���A��iC���A1p@�?�A��gB��@�S@�Cc@��A
��C�HC��8   "   �                           	                  
         1               D         !   U      3      r            8      *         F         
   ,   �            $         _      :      !   A                              #                        /               /            =      )      +            !      %         3            )                        '               %                                                      !               /            3                                       '            !                                 N��O�*NQ~�O"�N�%�Nˎ�N'wKNd�N��N�N#�!Ok��O�MPOq�BN�N�N���N�/VO�B4O��:N9N�Og�O$�	N��Pd}�N��N@�OfPo�N��O��M�UO�W�N�M�NR�?O5�OM�N�\~O�D�NABN��P�NgpkOh��N��O�b'O�qNc"�NJ��Op`�O��NXP7N�C�O��M�+Ok �N	[  �  	    I  �  a  ,  �  �    b  $  �  �    �  �  �  �  �  "  �  �  �  �  �  <    �  >  H  �    �  �  �  _  �  �  w  �  �  �    9  -  j  �  �  :  
:  _    �  $  p  �49X=�w�D���D��;D��<#�
;�o;�o;��
;ě�<t�<�o<u<�o<e`B<�o<�1<�t�<���=C�<��
<ě�<�1<�9X<�j<ě�<ě�=+=8Q�<�`B=49X<�=�C�=o=��=��=e`B=�w=<j=#�
=#�
=m�h=H�9=L��=e`B=��>o=ix�=m�h=�O�=�hs=�t�=���=��=�{=�;d=�������������  ���������
#/<CLQVWUH<#	���������������������a`cgqtt���������tngajkt���������vtjjjjjj#)*696//) vv��������vvvvvvvvvv���������������������������������������������������������������������������������
#(*++(#
 �� 	"/4;>EHB;/"	.,.<>HUafilnnhaU@</.�����������36BBO[a^[OB633333333����������������������������������������������������������
#/FKNPPLH<#()0<IJIE<0((((((((((������

������!#%)+/8<HUab_]YUH</!���������������������)4?E?5���##01<BA<30##040-#��������������������2//5BN_jp�����tg[NB2"#/;<FE@</#""""""&)BO[_fhhda[OB6+!������������������������������������������������������������XU[^gltztqg[XXXXXXXX����������������������������������������������������������&)59FNPQOONB5��������������������opx{���������{oooooo����)27:@=5)��#)"������������������������������������#&'% �������������	

�������������������������tst���������|ttttttt������)*11)���gglt����������{tpkggpqsz�����zpppppppppp������	

�������
)6BHNNLIB6)
vrzz������|zvvvvvvvv�����������������������������������˻��������ûǻûû������������������������5�A�^�i�k�g�a�Z�N�A�(����������(�5čĚĠĦİĦğĚĕčăćččččččččÓàìöùüùïìà×ÓÇÄ�z�t�s�zÇÓ�g�t�x��{�t�h�g�e�[�X�T�[�^�g�g�g�g�g�g�A�N�Z�f�g�r�s�t�s�i�g�e�Z�N�G�A�>�=�A�A�������������������������������������������������ɾ������������������������������N�U�Z�a�e�a�Z�T�N�A�?�7�A�G�N�N�N�N�N�N�������������׾־׾��������B�O�[�\�[�X�O�B�7�A�B�B�B�B�B�B�B�B�B�B�(�4�M�\�b�Y�L�A�4�(�������������(�H�T�a�g�m�r�q�m�e�a�H�;�/�$� � �)�/�=�H�����������������������}�s�j�n�s�t�|��<�A�H�L�H�>�<�/�+�*�/�:�<�<�<�<�<�<�<�<�z�~�����������z�x�t�s�x�z�z�z�z�z�z�z�z�����������������t�s�g�f�d�f�f�s�u���s�������s�n�f�Z�T�Z�^�f�j�s�s�s�s�s�s����������������ŹŭŠŔŎŋōŘŠŹ���ҿT�`�m�y�{�v�g�G�;�.�"��	� �����.�G�T���������������v�v�����������T�a�z�}���������z�x�m�a�[�H�@�:�A�H�L�T�����������������������������������������T�`�c�e�m�y�~�y�m�`�T�L�G�?�G�H�T�T�T�T����<�U�b�{ŉőœň�{�U�#�
����������������������������ߺֺҺ˺ֺغ����ּ�����ڼּҼռּּּּּּּּּּ������ʼμּݼּ̼ʼ����������������������"�T�`�e�a�T�H�;� �	��������������������������	��������������������������޾��ʾ����������׾ʾ��������������������*�6�C�K�C�6�*��������������ùϹܹ��� ������Ϲù����������������T�`�m�s�y���������y�q�m�j�`�V�U�T�R�T�T�m�y�����������y�o�m�g�h�m�m�m�m�m�m�m�m�;�H�S�T�`�a�m�a�\�T�H�;�/�-�*�+�/�5�;�;�h�tāčĚĠĦĪĬĦĥĚā�t�h�b�[�X�[�h���)�*�1�)�'������������������)�B�O�c�t�v�w�t�h�[�O�B�6�������)àìùúùôìàÓËÓ×àààààààà�������!�'�!���������������������Ƴ���������#�&������ƳƢƓƋƉƑƧƳ�������������������������������������������/�6�<�?�;�3�/�#��
���������������������������������������s�g�d�g�o�������5�A�K�N�M�J�A�5�(������������.�5D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D��y�{�����������y�p�p�r�x�y�y�y�y�y�y�y�y�F�S�V�X�S�J�F�>�:�7�9�/�:�D�F�F�F�F�F�F�)�5�B�N�\�g�t�g�Y�M�F�A�5�)�)ǔǗǡǬǪǡǜǔǈ�{�z�o�k�l�o�v�{ǈǓǔ�M�Y�f�r�f�`�Y�M�I�B�M�M�M�M�M�M�M�M�M�M�S�_�l�x��������x�l�_�S�O�L�S�S�S�S�S�S�ûлܻ�����������ܻϻ��������ý��!�"�.�7�.�!������������E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E{EvE�E�E*E7ECEPEREPEPECE7E6E*E*E*E*E*E*E*E*E*E* 1 1 4 8 < ? C S A M H G   & A � F k c ^ G + A \ ; 3 u ( J .  � ( 6 4 " 6 ^ " > h A A 6 U 4  a ^ n < D V % y $ `        V  ?  �  �  N  v  �  E  7  �    �  @  �  �  �  7  �  l  �  �  �  �  �  w  �  ?  �  s  �  c  �  c  &  �  �  �  W  �  �  �  �  8    >  �  �  `  ,  l  �  a  ]  �  O  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  E.  s  �    2  K  b  w  �  �  q  M  !  �  �  �    I    �  9  \  �  �  ;  �  �  �  �  	  �  �  �  y  n  3  �  2  G  �  �  z  �  �  �  �            �  �  �  �  M    �  �  k  !  9  I  C  /    �  �  �  �  g  0  �  �  j  '  �  �  m  1  �  �  ~  x  r  k  d  [  P  E  6  &      �  �  �  �  j  #   �  �  �  	  /  D  W  _  _  W  E  *    �  �  �  �  �  �  �  C  ,  +  *  )  "           �  �  �  �  �  �  �  l  a  V  L  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  �  �  �  }  w  p  i  `  W  N  ?  +    �  �  �  �  p  J  %       �  �  �  �  �  �  �  �  �  �  x  m  b  V  K  @  4  )  b  W  M  A  6  )      �  �  �  �  �  �  �  ~  f  E    �  �  �      $       �  �  �  �  �  �  T    �  J  �  -   d  ;  f  }  �  �  w  k  T  3    �  �  =  �  �  5  �    V   �  �  �  �  �  �  �  �  �  �  �  �  c  ?    �  �  Q    �  �      
          �  �  �  �  �  �    R  $  �  �  �  g  �  �  �  �  �  �  �  �  �  �  �  t  f  W  I  9  '      �  z  |  ~    �  �  �  �  y  i  T  ;       �  �  �  \     �  �  �  �  �  �  �  �  }  u  m  i  i  i  i  i  m  r  w  |  �  �  �  �  �  �  �  �  �  �  �  y  `  9  	  �  �  O  �  �  $  �  �  �  �  �  �  �  �  �  �  �  U    �  c  �  X  �  �  V  "  !  !                                %  )  .  o  }  �  �  �  w  i  X  A  %    �  �  p  1  �  �  O  �    �  �  �  �  �  �  �  �  }  q  f  [  S  J  9     �   �   �   �  �  �  �  �  �  �  �  �  �  �  |  n  a  X  Z  [  ]  ^  _  a  �  �  �  �  t  j  k  `  L  8  %    �  �  �  ]  �    �  :  �  �  �  �  �  }  j  U  @  +    �  �  �  �  s  O  *  �  #  <  9  7  4  1  /  ,  %        �  �  �  �  �  �  �  o  X  �  �       �  �  �  �  �  �  �  }  ^  3  �  �  T  �  �  �  l  �  �  �  �  �  �  �  w  8  �  �  ~    �  �  �  s  �  !  >  .    	  �  �  �  �  �  �  �  k  P  3    �  �  �  e    �  �  �    ,  A  H  @  6  (      �  �  �  U  �  |  �  6  �  �  �  �  �  �  �  �  �  �  �  �  �  p  \  H  6  $    �  
  
h  
�  �    _  |  }  l  G    �  #  
d  	�  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  q  j  d  ]  M  <  *    T  ^  v  �  �  �  �  �  �  s  _  F  )    �  �  l  3  �  �  z  �  �  �  �  �  �  �  r  `  M  8  !    �  �  �  ]  2  *  �  �  
  '  C  W  _  W  >    �  �  �  K    �  �  �  �  t  �  �  �  �  �  �  �    r  f  \  U  N  H  A  >  <  ;  9  8  �  �  �  �  �  �  �  �  �  e  )  �  �  `  "  �  �    �   �  w  d  P  7      �  �  �  �  �  �  �  �  �  �  �  �  T  �  �  �  �  s  d  T  A  +    �  �  �  �  k  C    �  v  �  �  �  �  �  �  �  �  �  �  �  c  ;  	  �  �  4  �  *  g  �  �  �  �  �  �  �  �  �  �  �  �  �  �  x  q  i  a  M  7     	    �  �  �  �  �  o  G    �  �  g  &  �  �  h    �  V   �  9  &      �  �  �  �  �  �  �  �  �  �  �  |  h  S  :     �  �    '  -  &    �  �  �    K    �  �  8  �  F  �  �  �    c  �  �    C  \  h  f  C  �  �  �  �  �  1  �  �  "  �  �  �  �  �  �  �  �  �    {  w  u  s  ~  �  �  �  �  �  �  X    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  :  #    �  �  �  �  �  �  x  O    �  �  S  �  �  N  �  �  
:  
  	�  	�  	k  	C  	  �  �  W    �  w    �  �      �  �  _  A  #    �  �  �  �  �  p  X  A  )     �   �   �   �   �   |    �  �  �  �  q  P  0    �  �  �  j  <    �  �  C  �  V  
�  $  Y  �  �  �  �  �  �  q  +  
�  
i  	�  �      �  Z  �  $        �  �  �  �  �  t  b  Q  ?  -      �  �  �  �  p  
�  
n  
Z  
H  
5  
  	�  	�  	�  	z  	:  �  w  �  /  P  N  �  d    �  �  `  #  �  �  k  (  �  �  n  )  �  �  (  �  �  �  c