CDF       
      obs    D   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�x���F       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       Pv��       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �+   max       =���       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�
=p��   max       @E���Q�     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�     max       @vE��R     
�  +|   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @Q�           �  6   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @�e�           6�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��j   max       >��       7�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�э   max       B,P�       8�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�r   max       B,@M       9�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >蹐   max       C��       :�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >�3�   max       C��       ;�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       =   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          7       >   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          3       ?$   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P/�a       @4   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�쿱[W?   max       ?��J�L�       AD   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �+   max       >�+       BT   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��G�{   max       @E��z�H     
�  Cd   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �������    max       @vA��R     
�  N   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @0         max       @P�           �  X�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @̄        max       @�L`           Y,   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         =�   max         =�       Z<   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?p�)^�	   max       ?��x���     @  [L   
                                 
   	                        C         6            	         '       M   ;   
         
   (      X   >   -      A                  9            "                     Q   �               
Nߖ�N~��O�7HN�tbOx��NK��O AxO	MO��XN-��N�[`N���Nu�bN:N#O]N0T�OҞM��OH47O�BPv��O�̎O�ePA^O}��N,��O�	NӚOO6�N�K�Of��O|�P&l�O�ىO 	:N�?N2�mN�Z	O�V�N`q�P#T�O��PBN�U9O��O���ObƨNƝWNc��N�%P.B$M�g�O��O��4O��O*//Ob~vN
��N��:Nn�N~��O�<�O��N���N��OO9�|O�NH7�+��o�u�e`B�e`B�T���49X�ě���o%   ;D��;��
;��
;��
;ě�;�`B<#�
<D��<D��<T��<T��<T��<T��<�o<�o<�o<�C�<�C�<�C�<�C�<���<�j<�j<ě�<���<���<���<���<���<���<�/<�`B<�<��<��<��<��=o=t�=t�=�P=�P=��=��=,1=@�=@�=H�9=L��=T��=T��=Y�=y�#=�%=�%=�1=�1=���lhkimt�������tllllll��������������������wuw����������������w����������������������������������������&).6BCOCB60)&&&&&&&&5336<ABOTY[]^[XOB;65.(&%0<?IMV]bb\UI<70.�������������������� ),)	        ��������������������!)05BHB@=53)"KNPY[gjqog[NKKKKKKKK/.0<?INKI<20////////��������������������

#%$#








#/<HVdgaXH@/*(" ^aampz}zma^^^^^^^^^^*+-/<HUalpnlaUSLH</*�������
"&�������)5BB<CVVNB)�� (5BN]dgiog[NB5) #26BO_jqqmh[@9)!���	"Halmi]TH;/	������������������������������������������)5N[diilkg[NB5b`^egtv|�����tlgbbbbWTU[ht���������tqh[W��������������������������������������������������

�����#/<HUZ\\_\H/��������������������������������������������������������������������������lkmrz���������zxpmll�������������������� #08;10#          �����%>B;1)���GBCJS]gt��������tgNG����
Uaz���na<#��������	�������!)-6?O[ht{}}{raO6)!���������������������������
#.'# 
����������������������htxwth[SR[hhhhhhhhhhEGHJO[hqhf`\[OEEEEEE26:Bgt��������tg[B62��������������������(&%$$(/;HSVZZ\TH;1/(iffmz�������������zi�������
"#"
����QTZ]bjmz�����zmgaTQ��������������������--/7<=A?</----------)/0050)��������������������&().)�������8BK@6)������������
 ! 
���vzz~�����������zvvvv\bfhtw������th\\\\\\������'(����+168:ABMO[bchqh[OB6+���������������������U�a�n�zÇÎÊÇ�z�n�a�U�L�I�U�U�U�U�U�U�����������������s�o�n�s�y������ĳĿ����������������������ĿĳĬĦĦĤĳ���������������������������������������	��"�$�"��	����������������������ÇËÓÚ×ÓÇ�z�z�v�zÃÇÇÇÇÇÇÇÇ�y��������������������������y�l�a�l�u�y�нݽ�����������ݽнĽý��������Ľͽл���	��+�4�0�'����ܻû����ûлܻ���f�g�n�g�f�`�Z�V�N�W�Z�e�f�f�f�f�f�f�f�f�a�f�m�o�m�l�e�a�V�T�H�G�G�H�T�V�a�a�a�a�)�*�6�?�B�D�I�B�:�6�)�����(�)�)�)�)�O�U�[�h�i�j�h�[�O�H�D�K�O�O�O�O�O�O�O�O�'�4�5�?�@�L�@�4�/�+�'�'�'�'�'�'�'�'�'�'�(�-�5�A�N�O�N�A�5�(��$�(�(�(�(�(�(�(�(�a�a�m�y�m�h�a�T�S�O�T�`�a�a�a�a�a�a�a�a�	��"�;�E�F�D�;�.�"���۾׾��ʾ׾��	�ĿѿѿҿѿɿĿ����ÿĿĿĿĿĿĿĿĿĿ��	��"�&�$�%�$�*�"�������������������	���Ŀѿ����� ����ݿѿĿ����������������
��<�IŊŠŞ�{�b�<���������������� �
�ʾ׾��������׾̾ʾ����������������ʾ��������¾ž�����f�Z�M�D�B�Z�f�p������;�T�a�i�j�g�Y�T�H�;�/��������"�*�5�;�����������������������������������������������������������������������������������������������������ưƫƵƷƻ�����h�uƁƎƓƒƎƁ�{�u�h�\�W�T�\�b�h�h�h�h�����
������
����������������������#�/�4�<�H�J�H�C�<�6�/�)�#����#�#�#�#ù��������������ùìàÓÇÃ�|�}ÇÓìù�ѿݿ��������������ݿҿο̿Ϳѿ����5�A�g�����������������N�������Ϲܹ��� �'�'���ܹù������������ù���*�6�=�C�C�J�N�C�6�*�����������"�.�6�;�G�N�J�G�;�.�"���	��	��������������������������������������������s�����������������|�s�g�Z�W�X�Z�g�m�s�s���������������������������������~�������ּ��������ּռԼּּּּּּּּּ��������������������������~�w�u�z�����ſĿѿݿ����0�@�3�(����ݿĿ�����������$�!�����'�/��������������� �������������������������������������������ܻ���'�4�:�<�:�4�'����лƻ����ûܿG�T�`�e�}���}�w�m�T�G�;�-�"����"�.�G�4�A�M�Z�f�s������������f�Z�M�;�4�-�-�4�y�����������������y�x�l�l�`�l�r�y�y�y�y���ݿѿ̿ȿ˿ѿݿ������������H�T�a�m�w�v�r�m�d�a�T�H�E�;�H�H�H�H�H�H�����������������������w�g�]�X�Y�d���������������������������~���������������
��#�-�6�-��
������������������ĦĳĿ������������ĿĵıĤĚĔďčďėĦ��)�0�5�B�N�V�Z�Y�N�B�5�)��������ŇœŠŭŹ��������žŹŭŠŘŋŊŇŁŃŇ�H�U�a�n�}ÇÈÂ�z�n�a�U�H�<�:�<�C�L�E�H�A�M�T�Z�\�Z�M�A�6�8�A�A�A�A�A�A�A�A�A�A¦²¶¾²¦�y�{�T�a�c�a�\�V�T�H�D�H�H�T�T�T�T�T�T�T�T�T�/�0�7�/�+�"���	���������	���"�.�/�/�����ּ������ڼԼʼ�������t�q�t�����DoD{D�D�D�D�D�D�D�D�D�D�D�D�D�D{DnDgDhDo�-�:�:�F�S�_�h�f�_�S�F�B�:�-�-�"�-�-�-�-�F�S�_�b�a�_�W�S�I�F�@�@�B�B�F�F�F�F�F�F�r�~���������������������~�r�e�]�Z�e�f�r�����������ɺ˺ɺɺ������������������������(�-�(��������������� H 6 H K C C F V Q p 7 U > u j 8 5 [ A > [ 4 N ? * \ & G M * L 6 a G ' X g 5 M . # ; � A L   F : J S E a L P $ D F J Y � � O $ y l 5 6 U    �  �  �    n  g  a  `  �  �  �  �  X  e  C  �  .  �  �  �      �  �  w  �  �  �  �  "    ^     W  �  �    �  i  �  6  �  �  H  ]  �  �  k  �  .  B  0  C  �  �  	  $  �  ]  �  �  (  �  �  �  Z  C��j�D��<o�o;ě����
;ě�<t�<��
;�o<�o<e`B<T��<o<o<D��=�P<e`B<���=o=��
=C�='�=�O�='�<�j=\)<���<ě�<�=ix�=aG�=���=��=\)<��<�=t�=��<��=�x�=�Q�=��P='�=\=0 �=m�h=8Q�=#�
=0 �=��=49X=u=�%=���=y�#=q��=T��=ix�=]/=m�h>	7L>��=��P=�\)=�l�=���=�;dB	�2B �@B�:B��B�{B�!B��B&|�B!��B�B�B?B��B&D'BBRB�^A�i�Bp�B�<B�B�B�A�эB�B��B�+B	�Bx�B�}B!�BTB�AB��BFBN}B" B �B;jB%vwB��B	�NBvwB�sB�tB!�B$SB,P�B�8BHB	/PB��A��B ѰB��A���B�`B��B�"BWpBp�Bg�B�B��B>�B��B�B�'B
:PB ��B�<B��B�B�gB�B&@B!��BFoB��B�SB	7�B&T�B?YB@�BA�A��rBc�B��B@jBNrB@A�rB@�B� B��B	��B��B�PB"=IBC<B�7BA�B?B:�B"1zA���B>�B%�B¤B	��B4�B�>B�0B �WB$?�B,@MB��B>�B	�B>�A��	B �	B��A�Z�B?bB��B?RBcQB�BGKB<IB��B6tBD�B�B��A�L�AEo!A�2A�3A�\8AɓZA��A*	@���A?�A�A��A��@�B�A��iA�� A["�Ay(sA���Ax�2A�N�AR#�AE�A�?�A�A�݂B �B��A��AgAˮ�A��A��1>蹐A���A`J@���A���A�#>AҔA��!AV�A��%@��?@�krAe��A?�2A,�A{��A��A�4<AG�A���A�{QA��A�πA�كA<IGA�yA��XA��@�xrC��@�7�@�;�@�l@ dA4vlA�{[AE��A�CA�gA���Aɂ�A��A*�3@��A>��A�x�Aև+A�'�@��+A��{A�~�A^��Ay�A���Av��A酌AR�"AH�A�}�A��]A�^VB^?Bs]A��8A�A˔A�yA�|>�3�A��Aaʄ@�Q$A�{�A��A	�A�E�A~�AҀ8@�@�b�AgKA@�tAD#A|]"A��A�{AH�DA��A�<A��A� �A�y�A<��A��]A�� A���@��-C��@@��a@
^@#��A4ԡ                  	                  
   	                        D         6            
         '   !   N   ;               (      X   ?   -      A                  :            "                     R   �                                          !                        #            7      #   %         !                  +   '                     )   #   /      %                  -                                 )   !                                                                  #            3      #   !         !                     !                                                   )                                                   Nߖ�N;ުO���N�tbNӬ�NK��O.N���O
�;N-��N��N���Nu�bN:N#O]N0T�OҞM��Nn�O�BP/�aOMҙO�eO�L�O/a\N,��O�	NӚOO6�N�:xOVbO->LOf^$O��+O 	:N�?N2�mN�Z	OcqN`q�O\�<O���Oa.4N�U9O�2�O���O8��N�3(Nc��N�%Pi"M�g�O�/O��4O.g3O*//Ob~vN
��N��:Nn�N~��O���OU]�N���N��OO~�O�NH7  I  �  7      '  �  �  �  �  _    �  �  .    �  �  �  �  b  #  �  o  C  W  p  �  �  �  W  C  	N  t  �  �  y  m  R  �  	�  |  �  E  	,  �  �  F  �  �  �  �  �  �  �    �  �  �  '  A        A  �  �  ��+�u�D���e`B�ě��T���t���o;��
%   ;�o;��
;��
;��
;ě�;�`B<#�
<D��<��
<T��<���<�o<T��<ě�<�1<�o<�C�<�C�<�C�<�t�<�/<�=ix�<�<���<���<���<���=C�<���=�hs=@�=8Q�<��=L��<��=C�=\)=t�=t�='�=�P=<j=��=H�9=@�=@�=H�9=L��=T��=T��=�7L>�+=�%=�%=�-=�1=���lhkimt�������tllllll��������������������|y����������������|����������������������������������������&).6BCOCB60)&&&&&&&&4568>BOQWZ[][WOIB?64/*(*09<IKSU[^WUI<90/�������������������� ),)	        ��������������������!)05BHB@=53)"KNPY[gjqog[NKKKKKKKK/.0<?INKI<20////////��������������������

#%$#








#/<HVdgaXH@/*(" ^aampz}zma^^^^^^^^^^9<<HUabbaWUHG<999999�������
"&�������)5;76>ONB5�##)5BJNY[`b_[NB5)##26BO_jqqmh[@9)!&;TaegcWPH;/ ����������������������������������������)5N[diilkg[NB5b`^egtv|�����tlgbbbbWTU[ht���������tqh[W���������������������������������������������������������"#',/<HLOOOMHC<4/)%"��������������������������������������������������������������������������lkmrz���������zxpmll�������������������� #08;10#          ������#(*+)��ONOSV[gt�������}tg[O!/7<HUantyznaUH</$#!�����	�������/-149BIO[istpnd[OB6/���������������������������
"#'%!
����������������������htxwth[SR[hhhhhhhhhhEGHJO[hqhf`\[OEEEEEE54;BNgt�������tg\D=5��������������������,-/4;BHMQTUTOH;530/,iffmz�������������zi������

����QTZ]bjmz�����zmgaTQ��������������������--/7<=A?</----------)/0050)��������������������&().)����!68?B@;0)�����������

������vzz~�����������zvvvv\bfhtw������th\\\\\\�����������+168:ABMO[bchqh[OB6+���������������������U�a�n�zÇÎÊÇ�z�n�a�U�L�I�U�U�U�U�U�U�������������s�r�p�s�|��������ĳĿ����������������������Ŀĳįĩĩħĳ��������������������������������������	������	����������������������ÇËÓÚ×ÓÇ�z�z�v�zÃÇÇÇÇÇÇÇÇ���������������������������y�n�l�f�l�y���нݽ�����������ݽڽнĽ��������ĽϽл���������"�%��������ܻܻ���f�g�n�g�f�`�Z�V�N�W�Z�e�f�f�f�f�f�f�f�f�a�d�m�n�m�k�c�a�Z�T�H�H�H�H�T�Z�a�a�a�a�)�*�6�?�B�D�I�B�:�6�)�����(�)�)�)�)�O�U�[�h�i�j�h�[�O�H�D�K�O�O�O�O�O�O�O�O�'�4�5�?�@�L�@�4�/�+�'�'�'�'�'�'�'�'�'�'�(�-�5�A�N�O�N�A�5�(��$�(�(�(�(�(�(�(�(�a�a�m�y�m�h�a�T�S�O�T�`�a�a�a�a�a�a�a�a�	��"�;�E�F�D�;�.�"���۾׾��ʾ׾��	�ĿѿѿҿѿɿĿ����ÿĿĿĿĿĿĿĿĿĿ��	�������	�����������	�	�	�	�	�	���Ŀѿ����� ����ݿѿĿ�����������������#�<�I�|ŌŔŌ�{�U�<�������������������ʾ׾����������߾׾ʾ����������������������¾ž�����f�Z�M�D�B�Z�f�p������;�H�T�a�e�a�X�R�L�H�;�/��	� � ���/�;�����������������������������������������������������������������������������������������������������ưƫƵƷƻ�����h�uƁƎƓƒƎƁ�{�u�h�\�W�T�\�b�h�h�h�h�����
������
����������������������/�1�<�H�B�<�4�/�*�#���#�)�/�/�/�/�/�/àìùÿ����ÿùìàÓÉÇÀÂÇÍÓÞà�ݿ���������������ݿؿӿҿٿ��Z�g�s���������s�Z�N�A�5�"����"�5�A�Z���ùϹܹ���������ܹù�������������*�6�=�C�C�J�N�C�6�*�����������"�.�6�;�G�N�J�G�;�.�"���	��	��������������������������������������������s�����������������|�s�g�Z�W�X�Z�g�m�s�s�����������������������������������������ּ��������ּռԼּּּּּּּּּ������������������������������������������ѿݿ�����������ݿѿǿĿ����ƿ�����������������������������������������������������������������������лܻ���� �'�+�3�4�'�����ڻлʻ˻пG�T�`�e�}���}�w�m�T�G�;�-�"����"�.�G�4�A�M�Z�f�s�~������}�s�f�Z�M�@�4�1�2�4�y���������������y�r�l�g�l�u�y�y�y�y�y�y���ݿѿ̿ȿ˿ѿݿ������������H�T�a�m�w�v�r�m�d�a�T�H�E�;�H�H�H�H�H�H���������������������������a�\�\�g���������������������������~���������
���#�)�#���
��������������������
ĦĳĿ������������ĿĵıĤĚĔďčďėĦ�)�5�:�B�N�Q�U�S�N�B�5�)��������)ŇœŠŭŹ��������žŹŭŠŘŋŊŇŁŃŇ�H�U�a�n�}ÇÈÂ�z�n�a�U�H�<�:�<�C�L�E�H�A�M�T�Z�\�Z�M�A�6�8�A�A�A�A�A�A�A�A�A�A¦²¶¾²¦�y�{�T�a�c�a�\�V�T�H�D�H�H�T�T�T�T�T�T�T�T�T�/�0�7�/�+�"���	���������	���"�.�/�/�����ʼּʼ�������������y�v�y���������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D{DzDzD{D�D��-�:�:�F�S�_�h�f�_�S�F�B�:�-�-�"�-�-�-�-�F�S�_�b�a�_�W�S�I�F�@�@�B�B�F�F�F�F�F�F�r�~�����������������~�r�m�e�_�\�e�j�r�r�����������ɺ˺ɺɺ������������������������(�-�(��������������� H 4 F K * C ? T 5 p 2 U > u j 8 5 [ L > g % N < ! \ & G M * ( ) Z B ' X g 5 E . * ' a A P   B = J S ; a M P & D F J Y � � -   y l ! 6 U    Z    �  �  n  5  #  4  �  �  �  �  X  e  C  �  .  �  �  �  �      m  w  �  �  �  �  J  r    �  W  �  �    �  i  �    �  �  J  ]  �  �  k  �  �  B  E  C  m  �  	  $  �  ]  �  �  �  �  �  F  Z  C  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  =�  I  C  =  7  0  (           �  �  �  �  �  �  �        �  �  �  �  �  �  �  �  �  �  �  �  �  n  U  8     �   �   �    ,  6  3  )       �  �  �  V  #  �  �  �  d  -  �  S   �    	      �  �  �  �  �  �  �  �  �  y  g  T  ?     �   �  `  �  �  �  �  �  �      �  �  �  �  �  �  j  K    �  �  '  &  %         �  �  �  �  �  �  f  9    �  �  z  G    {  �  �  �  o  \  G  ,    �  �  �  �  ^  5    �  �  �  �  {  �  �  �    s  b  O  :  !    �  �  �  N    
  �  �  ;  {  y  v  s  l  t  �  �  �  �  �  h  D    �  q    �    �  �  �  �  �  �  �  �  �  w  i  Y  G  5  $    �  �  �  �  }  P  [  _  ^  T  D  2      �  �  �  �  �  g  ;    �  �  Z    �  �  �  �  �  �  �  �  k  M  -  
  �  �  �  F  �  �  s  �  �  �  �  �  �  �  p  _  N  <  )      �  �  �  �  �  �  �  �  �  �  �  �  �    y  t  o  j  d  ^  W  O  H  @  9  1  .  #          �   �   �   �   �   �   �   �   �   �   �   �   �   �   �              �  �  �  �  �  �  �  �  �  �  �  v  f  W  �  �  m  F  $  	  �  �  �  �  �  }  b  B    �  �  V  �  }  �  �  �  v  k  `  U  I  >  3  )            �   �   �   �   �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ]  �  �  (  �  �  �  �  �  �  �  ]  0  �  �  �  r  7  �  �  �     �   )  �  
  ?  [  a  K  #  �  �  h  G  K  Q  G  �  �  (  f    �                �  �  �  �  t  T  7    �  �  �  g    �  �  �  �  �  �  r  f  }  }  {  ~  t  f  K    �  a  �  |   �  !  \  m  n  j  [  :    �  �  �  q  ?  �  �  N  �    �   �    &  7  >  B  :  )    �  �  �  �  �  �  j  ^  3  j  �  D  W  R  N  H  >  4  &      �  �  �  �  �  �  v  _  G  /    p  l  b  R  9    �  �  �  �  p  Y  F  2    �  �  u  (   �  �  �  �  �  �  �  |  s  g  ]  T  D  +    �  �  �  �  �  w  �  �  �  �  �  �  �  �  �  r  c  U  D  4  $    	   �   �   �  �  �  �  �  �  �  �  �  �  �  d  A    �  �  l  -  �  �  �  V    5  M  W  Q  <    �  �  a    �  g    �  Q  �  `  m  �    5  A  C  <  +    �  �  �  o  7  �  �    �  �  �   r  �  `  �  A  �  �  	  	;  	L  	K  	>  	"  �  �  )  �  �  X  �  �  J  q  t  p  h  P  7  .  "    �  �  Y  �  �  �  h  �  �    �  �  �  �  �  �  x  b  M  8  #    �  �  �  p  N  .     �  �  �  �  �  �  �  �  �  �  �  �  v  e  R  <  &     �   �   �  y  o  d  Z  O  D  9  -  !      �  �  �  �  �  �  �  �  �  m  k  h  \  N  8  !    �  �  �  �  �  p  S  3    �  �  a    ,  E  O  Q  K  ?  +    �  �  w  3  �  �  ,  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �    m    �  	  	Y  	�  	�  	�  	�  	�  	�  	�  	f  	  �  "  f  ^  �    )  �  �  #  S  s  {  q  `  I  .    �  �  <  �  "  �  �  _  �  6  ^  ~  �  �  �  �  �  �  r  i  ]  V  @     �  �    �  E  3       �  �  �  �  �  b  <    �  �  �  �  t  g  \  Q  �  �  �  	  	  	)  	*  	  	  �  �  s    �    k  �  �  (  �  �  �  �  z  n  b  T  D  4  !      �  �  �  �  �  �  {  X  �  �  �  �  �  �  �  x  b  J  /    �  �  y    �  8  �   �  .  8  ?  D  E  B  ;  ,    �  �  �  �  �  d  Q  S  A  '    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  f  X  I  ;  �  t  b  P  @  0  !      �  �  �  �  �  �  �  �  �  �  z  �  �  �  �  �  o  a  O  8    �  �  �  �  W  �  V  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  i  :  g  �  �  �  �  �  �  �  �  �  �  �  f  C  $    �  �  �  �  r  `  N  <  (    �  �  �  �  �  v  E    �  b  �  7  �  �  �  �  �  �  �  �  �  �  u  T  )  �  �  y    \  �  �   �    �  �  �  �  {  [  ;    �  �  �  y  A  	  �  �  w  a  n  �  �  �  �  �  �  �  |  ^  8    �  �  �  ^  3    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  o  ^  L  9  &    �  �  �  �  �  �  {  i  Y  H    �  �  '  !        
    �  �  �  �  �  �  �  �  �  �  �  �  �  A  2  $    
  �  �  �  �  �  �  �  �  �  �  g  N  1     �  
W  
          
�  
�  
�  
�  
:  	�  	}  �  V  �  �  �  <  �  2  �  �    f  �  �  
      �  m  �  5    z  �  �  �  	�    �  �  �  �  �  n  C  =  $  �  �  s  4  �  �  r  -  �  �  A  4  &      �  �  �  �  �  t  V  ;  !    �  �  �  �  �  �  �  �  �  �  �  z  S  "  �  �  P    �  d    �  �  �   �  �  �  �  �  �  �  o  ^  I  5  !      �  �  �  ]    �  �  �  �  �  u  Z  =    �  �  �  �  a  /  �  �  �  Y  !  �  �