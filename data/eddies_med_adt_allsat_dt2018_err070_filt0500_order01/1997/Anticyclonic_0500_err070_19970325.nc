CDF       
      obs    F   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�V�u       �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       MˤZ   max       P���       �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �t�   max       =ě�       �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�33333   max       @F������     
�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vc
=p��     
�  +�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P@           �  6�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�             7`   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ����   max       >��\       8x   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��   max       B4�       9�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A���   max       B44�       :�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       ?>�e   max       C��w       ;�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       ?R;�   max       C��n       <�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �       =�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          C       ?   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          !       @    
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       MˤZ   max       O��       A8   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�j   max       ?��,<���       BP   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �t�   max       >J       Ch   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>Ǯz�H   max       @F������     
�  D�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�=p��
    max       @vb�G�{     
�  Op   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P@           �  Z`   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @��        max       @�            Z�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Cb   max         Cb       \   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�䎊q�   max       ?���`A�8     �  ]   
                                             7            �         [         
         $         �   J                                          
   $      
   /   
                  B      =                        !      N��{O���O�*�N4��N���M�u@M��O[ZN�/�N-��OS��NU�NHN��N�ܒP$dN��sNC�O��_P��O4�O���P,�aNd%�N��N���OtV}O'��Os�N�&OHɳP���PMbO+y�N|��NN��N�sO�ON��cOE��N��O�!�N�T�O��hNv�yO"e�O=!3O�VSN���N���O��2N���N��aOE�PNE�UOw��O��LO�P�N�T	O�ءNu",N|PN��+N^$Ohs�N�svN�=�ON�rMˤZN%�ýt���j��o�t��D���o��o$�  $�  ;o;��
;��
;ě�;�`B<#�
<#�
<49X<D��<D��<T��<T��<�t�<�t�<��
<��
<�1<�j<�j<���<���<�/<�`B<�h<�h<�<�<�<�<�<��<��=o=o=+=+=\)=\)=t�=��='�='�='�=0 �=@�=H�9=L��=P�`=aG�=aG�=ix�=m�h=y�#=y�#=y�#=�C�=��P=�1=�{=�v�=ě�����������������������������������������3105BN[gt���{sg[NB93����������������������������������������	�������������������� �	"/8;@C;//"	��������������������!"#$/14/.#)/9=BH>5)BBBHOZ[^][OBBBBBBBBB)35<51)#����������������������������� ����������GIIFGIO[ht�������[OG&#$*6CDLNGC64*&&&&&&�������������������
#/<FQVUOH/# !

x������$!������x.-0257<IUbmljbUI<70.����
!"$#!
��������������������������������������������)3BOW[hjlh[OB6,)������  ������������{}�����������������{�����������������������������������������������BNW[gt������tg[NICBB���������6^Z6����6[t���|qhurhOB6)%ghgf`aht�������{ythg��������������������@>ABCN[gjga[NB@@@@@@	
 #**.//#
				%&'*-/3<HHNOKHB<</%%��������������������0-.45BR[fgorpg[NB;50st��������������wtssdaejmy�����������zmd)*/1<HJUaUSHD<1/))))������� �����������������������noionaUH<4<E@HUZakon������������uklv����mjnsz�������������zm++//<HJSUOH<2/++++++)-5=;51*)&$��������������������������������
#0;<=<0)#

#0:<ED<800#
�����������������������������������6?BLOOY[hkb[B6)~��������������������������������������������������������������������kiemtzz{zmkkkkkkkkkk#)35?BKEB?5)X[gpt������tg[XXXXXX������������������������

������	 �
"#++%#
						�����!%%&$��
����













����������������������黑�����û˻˻û����������x�n�q�l�f�l�z���U�a�n�xÁÅÃÂÄ��z�n�a�U�J�A�>�?�H�U����������������������������������y�������������������y�m�l�k�m�x�y�y�y�yù������������������ù÷ùùùùùùùù���"�#�����������������H�T�a�m�v�z�������z�w�m�a�T�H�G�?�=�G�H����)�6�@�6�5�*�)���������������������������������~�������������������ʾ׾�����������׾ľ��������������y���������������y�u�l�u�y�y�y�y�y�y�y�y�Z�f�h�j�f�e�Z�P�M�G�M�T�Z�Z�Z�Z�Z�Z�Z�Z���'�3�:�@�@�A�@�3�'�����������������	��������������������������߾A�M�s�����ʾ׾���׾ʾ��s�f�W�L�=�4�A�m�y�����������y�m�`�^�_�`�h�m�m�m�m�m�m���"�/�/�/�/�"�������������{�����~�q�`�G�;�.�"����"�.�G�T�`�m�{Óù��������������ìàÇ�W�H�A�H�W�a�zÓ���������������������������y�q�n�r�|��)�5�B�N�@�5�)�������������������)��5�b�x�}�g�]�\�g�s�y�Z�A�(�������������������������ÿÿ������������������#�$�'�/�5�6�/�-�&�#���������"�#����������������������ƽ���������������̾�����������������������}�z�s�r�t�������_�l�x�~�������������x�l�_�X�F�@�C�F�S�_���(�4�M�Z�f�i�h�k�k�M�A�4�(��������	�������	�����������������������������ݽٽݽ������@�Y�f���������������ּ������y�Y�>�7�@�ܼ�'�C�e�L�4���ܻû����������������ܼ�����������¼ͼͼʼ�����������r�h�g������������������������������������������y�����������������}�y�x�t�w�y�y�y�y�y�y���(�1�5�A�L�A�5�(�����������Z�g�s�������������������s�n�g�Z�Z�P�Z�Z�ּ�������߼ּʼ��������������ʼμ��O�[�h�t�w�|�t�q�h�f�[�O�B�=�>�>�?�B�F�O�6�7�?�B�M�J�B�=�6�1�)�"�����)�5�6�6�h�tāčđĚĠģĞĚĔā�t�q�k�i�e�\�a�h�������
����������������������������	�"�.�;�G�T�b�a�W�G�;�"��	�������	���(�)�*�(�"�������������àÓÇ�z�y�z�y�zÂÓâìùü÷ìäììà�������ܾ׾Ҿ;ƾ��������ʾ׾������������)�B�?�=�:�)�������������������D�D�EEEEED�D�D�D�D�D�D�D�D�D�D�D�D���������������ŹŰŭŠŖŠŭŹ���������������������������������������������������	���"�'�)�"��	�����������	�	�	�	�	�	�f�s�������������������~�z�s�k�f�`�]�f������ �(�&�!���������ۼ߼��～���������������������������������������������������������~�y�u�l�d�_�^�[�`�l���������ɺѺֺ�ɺ������������z�q�u�������s�������������������������s�Z�N�C�L�g�s����(�1�3�+�(�������� ������¿�������
�����
������¿³¥¥§ª¿ù��������������ùìéìíòùùùùùù�������������������������������������������	���"�#�"�"��	��������������������/�0�3�;�>�B�?�;�7�/�)�$�'�&�/�/�/�/�/�/����������������������������������������E�E�E�E�E�E�E�E�E�E�EuEsEtEuE{E~E�E�E�E�ǔǡǭǳǴǭǭǡǔǈǇǈǉǔǔǔǔǔǔǔ�0�<�I�O�U�]�[�U�J�<�0�#��
���
��"�0�3�2�3�@�G�L�S�L�@�3�3�3�3�3�3�3�3�3�3�3�H�<�0�#�����#�0�;�<�I�J�H�H�H�H�H�H > ' ? M # � e  > L 8 + 8 8  ; 9 H N % 2 > Q < n 1 1 A F 2 9 b Y c @ f Q V > = 3 @ 8 5 Z 8 e 0  [ ( - T * S \ P d  : H f 5 � F j E @ K v  �  	  &  w  �  I  L  �  �  _  �  `  ]    	    �  d  {  �  �  c    �  �  �  �  u  �  
  �     �  �  �  �  �  9  &  �  �  9  �  W  �  p  �  �  �  �  p  �  )  �  Q  H  }    �  �  �    �  �  �    �  �   �  �����;�`B<#�
�D��$�  <#�
;o<�/<e`B;�o<��
<���<49X<�t�<�9X=��<�o<�C�=�w>n�<���=0 �=�;d<�j=o<�=\)=@�=}�=o=C�>��\=��=L��=C�=o=0 �=<j='�=Y�=8Q�=u=D��=�o=t�=8Q�=49X=�hs=e`B=P�`=�9X=P�`=H�9=�7L=e`B=�O�=�O�=�=��P=��=���=�%=��=�+=�-=��=ȴ9==Ƨ�=��B!�B"j�B�B 	�B,�B0�B1	A��B��B�2B��B�VB��B��BYB�B0R5B�UBD�B6>B&ߙBpBJ�B�B�zBf5B�Bh~Bf�B#rB	?=BYBm�Be�B��Bd�B�B��B"�BAiB
��B ��B��B�sB+��B��B4�BY�B�	BǕBBB�:B%YB%:�B!��B,�,B��BvOB]1B�B!��A�̡B$�B	�B�B_�BJZB��B$rB:�B!�eB"K�BEB �B,�JBB"B3A���B��B�&BBB�CBŘB�2B<.B��B0H�B͡BNB?�B&��B@B?�Bf�B��BFqBT�BN�BC=B#;�B	�B?�B?�B��B�^BJ�B��B�#B"qB #B2B �&B�B�B+��B;�B44�B��B�rB��B?�B�B$�ZB%?�B!�"B,�rB�mB?�BB�B�^B!�UA�o�B#B	��B �B�-B?�BV�B$+�B/�@�Eg@�i�A���?>�eAn��AΓ5A��pA�x�A��A��AR A�)A?8x?��A��AH��Al�cA�kAf%}A�0{@�#�A���A���Au�LA��,B��AIc@���A:�$AY^&A/��@�X @���@�A�y2An��A�R�A� A {�A�_�A�]�A�!-A�!A`k�A�`AʰAS]	A�wKC�?YA��3A��gA�rADlA;UA!�A>�@��A�LTA�\A�B*A�ĊA�z�A���A�A�h�C��wB��A��?�4�A년@�֝@��1A�p�?R;�AmږA�tA�l�A�x�A�j~A�@ATc�A�A? h?�[hA�z�AIAk�MA�rAe
�A�v�@�A�y�A��\Au��A��cB��AJ @�C�A;kAYqA1�@� �@��@�4sA��;AoE�A���A���@�TA��A׀�A��0A�[]A`�A�"A�oAU �A�]C�E�A���A��A�c�AC�A�A!��A��@��A�WGA���A�ltA͒�A���A���A��A�hC��nB�A�gn?�OA꿘                                                8            �         \         
         %         �   K                                           
   $         0                     B      >                        !                                                      +            9      !   1                           C   9                                                                        %   %      !                           
                                                   !                                                !                                                                                                               
   N��{N��bO�*�N4��N���M�u@M��N�G�N?2EN-��O*��NU�NHN��N�ܒO�a�N��sNC�OMaO���O4�O��ZO@z�Nd%�N�gUN���O=(�N��O+G�N��OHɳO��O�`N�N|��NN��N�sO�ON��cOE��N��Or�lNt��Ov��N2SO"e�O=!3O�VSN���N���OhG�N���N��aOE�PNE�UN��Oz�ROn<5N��O�7N,�9N|PN��+N^$Ohs�N��N�=�OE�MˤZN%��  [  �  �  �  q  [  �  m  �  �  �  e  h  �  �  �  �  5  c  =  �  �  
H    �    �  X  �  �    �  3  %  	  :  �  q  �  X  �  �  �  �  �  E  ]  �    �  F  ,  R  �  �      
�    	�  �  �  D  �    k     �  �  ]�t���`B��o�t��D���o��o<49X;�o;o;�`B;��
;ě�;�`B<#�
<�`B<49X<D��<���=��
<T��<�1=y�#<��
<�9X<�1<���<�/=+<�/<�/>J=���=o<�<�<�<�<�<��<��=C�=t�=��=C�=\)=\)=t�=��='�=T��='�=0 �=@�=H�9=m�h=T��=�hs=u=��=y�#=y�#=y�#=y�#=�C�=���=�1=� �=�v�=ě�����������������������������������������3105BN[gt���{sg[NB93����������������������������������������	��������������������	"/57/."		��������������������!"#$/14/.#)-56;=A75)BBBHOZ[^][OBBBBBBBBB)35<51)#����������������������������� ����������NNQV[ht�������th[UON&#$*6CDLNGC64*&&&&&&�������������������#/9<BHNOMH</#��������������������.-0257<IUbmljbUI<70.���������
�������������������������������������������0*6:BOS[hhih[OB60000������  ����������������������������������������������������������������������� ���������BNW[gt������tg[NICBB������� #!�����32366@BORYZYWUOIB763dhjt�������xutskkihd��������������������@>ABCN[gjga[NB@@@@@@	
 #**.//#
				%&'*-/3<HHNOKHB<</%%��������������������0-.45BR[fgorpg[NB;50st��������������wtsshdgl{������������zmh,-/7<BHUMH</,,,,,,,,������������������������������noionaUH<4<E@HUZakon������������uklv����mjnsz�������������zm++//<HJSUOH<2/++++++)-5=;51*)&$��������������������������������
#0;<=<0)#

#0:<ED<800#
����������������������������������������
68@BOX[hj`[B6)
������������������������������������������������������������������������kiemtzz{zmkkkkkkkkkk#)35?BKEB?5)X[gpt������tg[XXXXXX����������������������

��������	 �
"#++%#
						�����!$$%!��
����













����������������������黑�������������������������~�������������U�a�n�xÁÅÃÂÄ��z�n�a�U�J�A�>�?�H�U����������������������������������y�������������������y�m�l�k�m�x�y�y�y�yù������������������ù÷ùùùùùùùù���"�#�����������������a�f�m�r�u�s�m�a�U�T�M�H�T�U�a�a�a�a�a�a���)�6�8�6�-�)�����������������������������������~�����������������ʾ׾޾���������׾ʾ��������������ʽy���������������y�u�l�u�y�y�y�y�y�y�y�y�Z�f�h�j�f�e�Z�P�M�G�M�T�Z�Z�Z�Z�Z�Z�Z�Z���'�3�:�@�@�A�@�3�'�����������������	��������������������������߾������ʾоվо�������s�f�^�X�W�`�s�����m�y�����������y�m�`�^�_�`�h�m�m�m�m�m�m���"�/�/�/�/�"�������������T�`�m�r�y�{�w�m�k�`�U�G�;�.�(�$�'�7�G�TÓìù��������øìàÓÇ�w�q�n�p�v�zÇÓ���������������������������y�q�n�r�|�������)�5�B�I�C�5�)�����������������(�5�A�F�N�Z�\�V�N�A�5�(�#������!�(�������������������ÿÿ�������������������#�-�/�2�3�/�,�#�#��������������������������������ƽ���������������̾��������������������������w�y��������_�l�x�|����x�l�l�_�^�S�F�F�F�H�S�V�_�_�4�A�M�Z�b�c�e�Z�X�M�A�4�0�(�����(�4���	�������	�������������������������������ݽٽݽ����������ʼ׼����ּʼ��������������������ܻ��������������ܻջлʻлѻܼ������ǼǼ�������������r�q�r��������������������������������������������������y�����������������}�y�x�t�w�y�y�y�y�y�y���(�1�5�A�L�A�5�(�����������Z�g�s�������������������s�n�g�Z�Z�P�Z�Z�ּ�������߼ּʼ��������������ʼμ��O�[�h�t�w�|�t�q�h�f�[�O�B�=�>�>�?�B�F�O�6�7�?�B�M�J�B�=�6�1�)�"�����)�5�6�6�h�tāčĚğĢĝĚĒā�t�s�m�j�g�b�^�d�h������������������������������������"�.�;�G�M�[�\�T�Q�G�;�!��	�������	��"���$�(�(�(��������������àÓÇ�z�y�z�y�zÂÓâìùü÷ìäììà�������ܾ׾Ҿ;ƾ��������ʾ׾������������)�B�?�=�:�)�������������������D�D�EEEEED�D�D�D�D�D�D�D�D�D�D�D�D���������������ŹŰŭŠŖŠŭŹ���������������������������������������������������	���"�'�)�"��	�����������	�	�	�	�	�	�f�s�������������������~�z�s�k�f�`�]�f������ �(�&�!���������ۼ߼��～�������������������������������������������������������~�y�u�l�k�l�k�l�y�z���������������ɺϺȺ������������{�r�w�������s���������������������������s�n�g�^�e�s���$�(�.�(�'�����������������������������
������¿¯¬­±¿��ù��������ýùìììïöùùùùùùùù�������������������������������������������	���"�#�"�"��	��������������������/�0�3�;�>�B�?�;�7�/�)�$�'�&�/�/�/�/�/�/����������������������������������������E�E�E�E�E�E�E�E�EuEsEuEuE{EE�E�E�E�E�E�ǔǡǭǳǴǭǭǡǔǈǇǈǉǔǔǔǔǔǔǔ�0�<�I�N�U�\�[�U�I�<�0�#��
���
��"�0�3�2�3�@�G�L�S�L�@�3�3�3�3�3�3�3�3�3�3�3�H�<�0�#�����#�0�;�<�I�J�H�H�H�H�H�H >  ? M # � e 8 C L : + 8 8  8 9 H H  2 A  < d 1 # % = 5 9  D ^ @ f Q V > = 3 @ 4 1 , 8 e 0  [  - T * S E K W " 9 M f 5 � F f E > K v  �  �  &  w  �  I  L  �  a  _  i  `  ]    	  �  �  d  �  i  �    �  �  "  �  �    n  �  �  �  9  I  �  �  �  9  &  �  �    �  �  9  p  �  �  �  �  �  �  )  �  Q  
  B    �  �  Q    �  �  �  �  �  �   �  �  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  Cb  [  R  H  ;  .      �  �  �  �        �  �  �  t  A    E  f  �  �  �  �  �  �  �  �  �  �  �  t  A  �  �  H  �  d  �  �  �  �  �  �  q  D    �  �  n  5    �  �  P  �  l  ~  �  �  �  �  �  �  �  �  �  �  {  p  d  X  K  ?  5  9  =  A  q  k  e  ^  X  Q  K  E  >  8  2  +  %         �   �   �   �  [  W  �  _  ]  4  
  �  �  �  Z  -  �  �  �  m  ;    �  �  �  �  �  �  �                  %  '  !            3  �  �    :  T  b  k  k  b  M  -    �  l  �  v  �  a  �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  P  #  �  �  �  y  �  �  �  �  �  �  �  �  �  �  }  y  t  p  l  h  d  _  [  W  �  �  �  �  �  �  �  �  q  `  J  3      �  �  �  �  �  �  e  Y  J  8       �  �  �  }  V  -    �  �  �  V  '  �  �  h  g  f  e  b  T  G  9  (    �  �  �  �  �  �  l  C     �  �  �  u  d  R  ?  *    �  �  �  �  m  J  (  
  �  �  �  y  �  �  �  �  �  �  �  �  �  �  z  y  g  Q  1    �  �  �  v  ^  �  �  �  �  �  �  �  �  �  �  ~  G    �  N  �    b  �  �  �  �  �  �  �  �  |  s  h  ]  S  G  :  .  !    
   �   �  5  ,  #      �  �  �  �  �  �  �  �  r  b  S  C  4  %    2  M  [  _  a  c  \  M  9    �  �  �  F  �  �  R  �  +  @  /  �  Q  �  	M  	�  
;  
�  
�  +  <    
�  
�  
7  	|  _    X  �  �  �  �  �  �  �  �  �  �  x  \  >    �  �  �  �  }  ~  �  �  �  �  �  �  �  �  �  o  [  O  C  1    �  �  �  +  �  }  b  �  G  	)  	t  	�  	�  
   
@  
H  
3  	�  	�  	  �  �    �  �  A    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  t  g  Z  M  z  �  �  �  �  �  �  �  y  d  M  4      �  �  �  �  �  �        �  �  �  �  �  �  �  l  J     �  �  �  Q    �  �  �  �  �  �  �  �  �  �  �  �  �  v  W  6    �  �  �  u    �    J  X  L  6       �  �  �  �  S    �  b    �  1     U  �  �  �  �  �  �  �  �  v  S  %  �  �  f    �  d  �  �  �  �  �  �  �  �  �  �  x  o  e  W  I  <  0  $       �   �      �  �  �  �  �  �  �  �  �  �  �  �  x  X  7     �   �  �    �  t  �  �  N  �  �  �  q  �    C  N  5  �  
�  �  �  �  �  �  k  �  �  �  �      �  	  1    �  w  �  �  "  	  �  
    $         �  �  �  �  �  �  �  �  z  d  \  a  �  	       �  �  �  �  �  �  �  �  �  x  `  H     �   �   �   N  :  1  (          �  �  �  �  �  �  �  �  �  �  �  �  ~  �  �  �  �  �  �  �  �  �  �  x  `  H  0    �  �  �  �  �  q  M  (     �  �    K  (    	  �  �  ~  0  �  }    �  '  �  p  a  h  d  X  L  A  6  *        �  �  �  �  �  x  U  X  G  4      �  �  �  h  4  �  �  �  A  �  �  D  �  |  
  �  �  �  �  �  �  i  M  +    �  �  �  g  (  �  �  c    �  �  �  �  �  �  �  i  G  %    �  �  m  +  �  �    �    �  �  �  �  �  �  �  �  �  �  �  ]  3    �  �  w  N  )    �  �  �  �  �  �  �  �  �  �  o  L  $  �  �  �  F  �  �    O  �  �  �  �  �  �  �  �    v  l  c  Z  N  @  1  "       �  E  ,    !  &    �  �  �  �  �  �  �  �  �  �  �  |  a  E  ]  V  P  E  :  ,        �  �  �  �  �  {  ^  =       �  �  �  �  �  �  �  �  f  E  '    �  �  �  �  h    �  k      �  �  �  p  H    �  �  �  L    �  Y     �  F  �  .  �  �  �  �  p  X  ?  &    �  �  �  o  5  �  �  a    �  �  �  �    *  9  B  E  7    �  �  �  `    �  Q  �  H  �  �  W  ,         �  �  �  �  �  �  i  N  3    �  �  �  �  �  �  R  D  6  (         �  �  �  �  �  �  �  {  r  c  8     �  �  �  �  �  �  �  �  �  t  U  2    �  �  �  Q    �  i   �  �  o  ^  M  9  &    �  �  �  �  �  g  I  )  
  �  �  �  �  �  �  �  �                  �  �  �  �  j  �  G   �           �  �  �  �  �  t  Z  >    �  �  �  �  �  (  i  	�  
3  
a  
w  
�  
�  
�  
l  
H  
  	�  	�  	6  �  =  �  �  �  =  �                  �  �  �  �  Y  (  �  �  C  �  �  T  	a  	�  	�  	�  	�  	o  	M  	*  	  �  �  �  X  �  s  �  �  �  "   �  �  �  �  �  �  �  �  �  �  �  �  v  S  /    �  �  �  l  G  �  �  �  x  p  h  `  Y  Q  I  8      �  �  �  �  �  o  V  D  1      �  �  �  �  x  W  5    �  �  �  m  D    �  �  �  �  �  �  |  i  W  E  0    �  �  �  �  �  x  c  O  ;  (    �  �  �  g  <    �  �  �  h  ?    �  �  �  p  )  �    j  j  b  Y  B  !  �  �  �  g  1  �  �  m    �  0  �    ~       �  �  �  i  ;    �  �  �  x  H    �  �    �  '   �  �  �  t  Z  <    �  �  �  C  �  �  \    �  O  �  �  3    �  �  �  �  �  t  p  k  f  b  [  S  L  D  <  0  #      �  ]  M  =  -        �  �  �  �  �  �  �  �  v  r  p  m  k