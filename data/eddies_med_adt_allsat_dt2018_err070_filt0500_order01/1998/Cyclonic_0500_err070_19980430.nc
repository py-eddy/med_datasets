CDF       
      obs    K   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�������     ,  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M���   max       P�E�     ,  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��E�   max       <ě�     ,      effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?
=p��
   max       @FE�Q�     �  !0   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ����R    max       @vlQ��     �  ,�   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @-         max       @P�           �  8�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @Ϥ        max       @�F�         ,  98   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �C�   max       �D��     ,  :d   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�   max       B0LD     ,  ;�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�y�   max       B0B5     ,  <�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >��   max       C�.�     ,  =�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >��k   max       C�)�     ,  ?   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          e     ,  @@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          ;     ,  Al   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9     ,  B�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M���   max       P��P     ,  C�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��4֡a�   max       ?��G�z�     ,  D�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��E�   max       :�o     ,  F   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @?�����   max       @F/\(�     �  GH   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���\)    max       @vj�\(��     �  S    speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @%         max       @P            �  ^�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @Ϥ        max       @�ܠ         ,  _P   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         D�   max         D�     ,  `|   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?y�_o�    max       ?��G�z�     p  a�   d   .         1                     $   !      	         ;   
   4         1       2   5         -         G                              "                                    	                            	      $            F      
      &      	Pr\�O9�O3|mO֮PE�-O���N��N�sOJV�O�XN5��O��O���O���N)��N0��N?)�O��?O?�uP��Ogo�N�_�P��Os�PJ�P�E�OL��N?1nO�EIN�N�Pe�P{H�N??�N~O/�O��N�y�O�$N���OH�mN��O� �Om��Olq�Nn�NEĿOD��O�k�P�NS�N(`HO07�N}h�N��N���Ou/4O#s�O*�GN���N���N"��O�N��?O
��O���N[iN��ZO��OqD�M���NbT�N�b*OAZXN�D%N?*B<ě�;o:�o%   ��o���
�ě��o�#�
�49X�D���D���D���e`B�u��o��o��C���C���C���t���t���t����㼣�
���
��1��9X��j��j�ě����ͼ�����/��/��/�����C��C��\)�\)�\)�t��t��t���w��w��w�#�
�#�
�',1�,1�49X�8Q�<j�@��D���H�9�P�`�aG��m�h�}�}󶽁%��%��+��7L��t����P���P���P��{��E�����)BWd`ZNB)������������������������04?QURNKI<90'�/6BOUVTB6)���)BNadfp[B)�����������������������������	�����gnz����zncgggggggggg����
##%#������������������������aamxz{zrmcaaaaaaaaaa������	�������������������������������6CKO[[fe\WL6*��������������������<BN[]\[NBA<<<<<<<<<<,/2<HIIHE<50/,,,,,,,"&0;HTakrz}zmaT;-! "��������������������2>BN[gt�������tgM;52�)5?MIB>5-)����258BENSY[a[WNBBB5222&6BOZht����owh[UK7!&/U\`^`UPD</-%#�������������������#:Ib������nbL0��������������������������������������������������������������� #$��8B[kt���snWRSOB33/08U]`an���������na`\QU������� ������������hhht{���thd\hhhhhhhh����������������������������������������45BNP[]agkgb[NLB=544	')*6763)#7<DHUZa]UUKHH<;81077knz���������ztnjfceklmz��������zymkhllll����������������}�
#/HUaglkaUH</)#
��*.5<@?5'�������������������������������������������BMOV[hhpxtoolh[UOB8B��
#+8<C<<60#
���Rb{����������sg^VTIR)6??6)�����������������������������������������������������������������������������������)069?60)��")5BNRY\_^`[NB5)$! "xz�����������zqoqtxx�������� ����������BHT[abbba^[UTTNHEA>B������������������������������������mnz������������zrnmm��������������������"#/<CFEA><9/#"  ""������)6AEFB6������������������������
 !
���������������)5<;:6-) ���������������������558BNTSQNB:555555555LNQ[gkkiiiigf[NLEHLLY[\cgpt���������g[WY &(   �������������������������j�b�Z�Q�N�s����������������������������������������!�.�6�6�5�/�.�!����������������ʼּ������Ӽʼ��������a�T�H�;�"��	�����"�/�H�e�t�w�x�n�m�a���|�u�q�q�s��������� ���ڿѿĿ������0�(��#�-�<�I�U�b�n�{�{�r�p�k�b�U�I�<�0�������׾ʾʾʾо׾����������������H�G�C�D�H�P�U�Y�Y�U�H�H�H�H�H�H�H�H�H�H����ŶŶŲŬŭŹ���������������������������������������������%�'� ���������)�)��)�/�6�8�B�F�B�6�5�)�)�)�)�)�)�)�)���پԾҾԾ����"�2�;�B�C�;�.�"�����T�?�5�3�>�C�G�T�`�y���������������y�`�T�"�	���Ծž������ʾ����	��%�+�/�4�.�"�r�m�g�r�������������x�r�r�r�r�r�r�r�r�f�^�]�d�f�s�u�|�u�s�f�f�f�f�f�f�f�f�f�f�
��
����#�&�/�5�/�#�!��
�
�
�
�
�
��������ĿĽ����������������
��������ݿٿѿο˿ѿݿ߿������������׾����������ʾ;Ӿ��	�"�0�+�'������¤¦²¿������������¾¶²¦Ň��{�p�n�k�n�{ŃŇŊŔŜŠŠŠŚŔŇŇ�S�>�7�9�F�h�s�����������лۻջû����l�S�6�/�)�%�&�)�6�B�O�S�[�h�n�j�h�[�Y�O�B�6�z�u�sÁÓù����������������ùìáÓÇ�z�����q�N�@�3�#�5�_�����������������������T�L�@�9�6�;�H�S�T�a�m�q�z��~�z�r�m�a�T�4�/�(�0�4�9�A�J�M�P�R�M�C�A�4�4�4�4�4�4�ɺ��������ɺֺ������,�9�4����ֺɺɺź��źɺֺ�������ֺɺɺɺɺɺɾ��޾ʾ����ʾ��"�G�`�y�����m�T�;�	��ȹ������x����¹ܹ���'�A�a�e�_�3���ȿݿӿѿĿ��ÿĿѿ׿ݿ��ݿݿݿݿݿݿݿݼ��������������������ܻۻлѻܻ����������������ܻܽн̽Ľ������Ľнݽ�����������ݽ��V�R�N�S�V�`�b�o�x�{�~�{�x�p�o�f�b�X�V�V�r�f�X�M�K�A�=�A�U�Z�d�s������������t�r�M�I�A�>�>�A�M�Z�f�g�s������s�f�Z�M�M�Z�W�L�N�O�Z�g�s�������������������s�g�ZƳƱƪƭƳƼ��������������������ƳƳƳƳ�����������������$�0�9�;�7�0�$�������"����#�,�8�;�H�T�\�a�]�W�M�H�<�3�/�"�w�z�m�h�m�z�����������������������{�z�wŠŘŔŇńŇŔşŠšŭŵŹŹŹŭŠŠŠŠ�b�`�U�I�H�I�U�b�n�{�~�~�{�z�n�d�b�b�b�b�ù��������������ùϹܹ�����������ܹûܻл����������ûһܻ����������ܻû������л����4�M�]�\�U�@�'����л����������������������������������������޻l�d�_�[�_�a�l�q�x�y�y�x�l�l�l�l�l�l�l�l�B�=�A�E�M�O�[�h�tāčĔďčā��t�a�O�B�3�0�1�3�>�@�L�Y�]�Y�W�L�I�@�3�3�3�3�3�3�L�K�H�F�H�L�Y�e�i�m�g�e�Z�Y�L�L�L�L�L�L���}���������������������������������������
���*�6�C�O�S�U�T�O�H�C�6�*��"� ���"�/�1�;�H�K�S�T�Y�`�T�H�;�/�"�"����ĿĸĳĳĺĿ����������������������������������(�-�5�A�N�S�N�A�5�(�������������������
���������������������
�
���� �������������������������Ŀƿѿݿ���ݿܿԿѿʿĿ�ƎƉƎƖƚƧƳ��������ƳƧƚƎƎƎƎƎƎE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��4�.�4�9�4�'����'�4�@�F�L�P�S�Y�M�@�4�g�a�b�e�g�s�w��������s�g�g�g�g�g�g�g�gE*E$E*E+E7ECEPE\E`EhE\EPECE7E*E*E*E*E*E*àÓÒÊÇÄÅÃÇÓ×àâåìôõôìàĳĦĚĖđĐĐĒĚĦĳĿ������������Ŀĳ����������������������������������������a�]�U�M�R�U�a�n�r�w�n�d�a�a�a�a�a�a�a�a�h�`�[�Q�S�[�h�tāčĚěĚĔďčā�t�h�h�<�6�/�#�������#�/�<�H�P�R�R�P�H�<�y�q�y�������������������������������y�y�������������ûлڻл˻û��������������� 8 ( m H 7 : h g 7 J 3 $ 6 h k > [ & F G ^ * Y 8 V N ( 7 . &  Z H r ' $ � G D " !  n t Q p 3 ) ^ @ S K ^ V u $ 4 . h d \ J e E q A Z w   ~ $ L ( � F    �  �  �    Z  t  �  \  �  N  D  �  ^  �  ^  _  z  �  �  �     �  �    �  �  �  X  >  �  �  [  a  _  1  I    �     �  �  A  G  q  �  �  �  �  �  w  \  �  �  �  e  �  ]  v  ^  �  n    �  B  �  p  �  �  �  [  h  .  �  �  X����49X�D�����ͽY��+�e`B�u��j��㼋C��D���49X�C��ě����
���㽛���/��C��t�����7L�L�ͽ�\)����0 ż�����C��'D���ě���`B�o�@��0 Žt��D���<j�y�#�'�C��@��P�`�0 Ž0 Ž�7L��C��y�#�ixս<j�}�Y��P�`�P�`���-��t��}�u�aG��ixս�\)��+��1�ȴ9��O߽��w��{�C����-��1��j��`B��E��Ƨ�B
BZ7B%��BSTBVB�B �BZBx2B��A�v'B*�B+,�B0LDB +OB)`B�
A�B��B	�dBB�B�RBܖB�mB?�B&��B:?B!B�Be�B<bBT�B>�B�QB%�B��B"�B�DB�DBLRBj1A��>BoSB��Bn	B)�B@�B=�B$�B(�ZBb%B!��B	�B��B!��B�zB�B Q�BF�A�%:B��B��B�B\�B,�B�RB��B��B'B�B
�=B�JB�B
aBV�B9EB�\B@hB&D�BA�B��B7EB?�B>B�GB�A��B?�B*��B0B5B @5B2�B�A�y�BRjB	�jB�)B�CBOB�%B>�B'LB��B!AVBC�BA�B��B��B�/B�B�FB!B;jB�SB�B@
B )�Bn>BB��B>LB@"B?�B$8�B)=+BA�B!��BA|B��B"?FB8�B��B @�B?�A��FB��B�WB<�B>|B@B�7B�YB;RBpMBB
�GB�+B��B
?SB��B@�A��A
Uv@���A�6&Av�xA���AUc]A��A�?�A�*�A�AKAZ�	Ai�fAYU�@��sAA��A�ܡA�*pA�N5AY�A��A�j�@��AحzA��A��A�!wA:s�@L�V@>B�A\�?��A{b�@�:5@�jA+:@B:-ABC�A>ħA���B�>B	�A�vA��A��A��>��@�f�@���A��@�"A�W�?��?Ҋ�A��+A��xA���A�)�A�֚A/�B�>Ay��B�C�.�@�F�A�g7C���A�|MA�g�A��OA�*A��DA���A��@��A��A	�@��sA���At�vA��DAS�[A�b�A��A��&A�6�AY(AiNYAY�~@�}AA3�A���A��A�|	AY�WA�w>A�}@��4A؞�A̅|A�}3A�l�A:��@H��@<o�A\�h?/�A{|c@�e@���A*(pB?�AC
/A=aA�qFB�mB	4�A��(A���A���A�e�>��k@�;@�NJAЋ�@�/NA�|d?��q?� RA�Z�A�¨A���A��A��A1"�B	@�Ay,Bv�C�)�@���A���C���A�i�A�|�A�uA�w�A�u|ArA#�@���   e   /         2      	               %   !      
         <      4         2       3   6         .         H                              "                                    
      !                     	      %            F            '      	   1         #   +               #         %   #                  )         /      )   ;         %      3   7                                    !            #   /                                                !                                             '               #                                       #         9         %      #   9                                                   /                                                                              O���O�O3|mO��P��O�g<N��N�sOJV�O�sAN5��Omx�Oj9�O'�N)��N0��N?)�O��O?�uO]Ogo�N���O�r�O#��Oc��P��PO1v N?1nO�P0N�N�O��Pq�rN??�N~O/�O�@N2��O��N�t�O.��N��O]daOm��OPNn�NEĿN~��O���P�NS�N(`HO#VONT)=N��N���O0��O��O�N��SN���N"��O�N��?N�6�O��N[iN��ZO�7O8�gM���NbT�N�b*OeN�D%N?*B  	_  
  '  �  �  b  0    i  �  r    �  ^  $  '    
8  W  s    �  �  �  V  �  6  �  �  F  �  �  (    �  �  �  #       ,  C    @      �  �  �  D  �  �  �  ,  �  0  �  �  �  _  C  +  p  �  t  '  z  l  �    �    �  �  ��ě����
:�o���
�u�ě��ě��o�#�
�D���D���ě��ě����
�u��o��o���㼋C��8Q켓t�����������0 ż�1��j��9X��/��j����/������/��/��`B�o�o�t��t��\)�,1�\)��P�t��t��Y��<j��w�#�
�#�
�,1�0 Ž,1�49X�P�`�D���D���H�9�H�9�P�`�aG��m�h��O߽�t���%��%��7L���w��t����P���P���置{��E����)5=EHEB6������������������������04?QURNKI<90')6BOQNHB6)
	.BN[_a_[N5-�������������������������	�����gnz����zncgggggggggg����
##%#��������������� ����������aamxz{zrmcaaaaaaaaaa��������� ������������������������������!*26>COPTUTNC6*��������������������<BN[]\[NBA<<<<<<<<<<,/2<HIIHE<50/,,,,,,,!#'1;HTajqz|zmaT;/#!��������������������Y[gptw��������thg[VY�)5?MIB>5-)����35:BINQXZUNB53333333(-6BOZcht~{n[OB>/)(#)/<@HTTQIH<:/*#���������������������#;Ib{�����nbK0��������������������������������������������������������������� #$��46:B`nt���sh_OB96354U^enz���������naa]QU������� ������������hhht{���thd\hhhhhhhh����������������������������������������55BNO[[^[NNB?5555555)/6662+)"
:<FHUW_[UPH<52::::::lnz|���������znkgefllmz��������zymkhllll��������������������
#/HUaglkaUH</)#
�),5;??=5(�����������������������������������������LO[[\hmkh`[OLILLLLLL�
#.32.+*
������Rb{����������sg^VTIR)6??6)�����������������������������������������������������������������������������������)069?60)��(05BLNTXZYONB>5)(%$(yz�����������zspruyy�������������������CHTZaaaaa]YTPIHEB?CC������������������������������������mnz������������zrnmm��������������������##/:<=<:7/##"!######��)6:?<6)�����������������������
 !
��������������)1551)'�����������������������558BNTSQNB:555555555LNQ[gkkiiiigf[NLEHLLagit��������tgg`aaaa &(   �������������������������~�u�r�r�s�y�������������������������������������!�'�.�/�.�.�)�!�������������������ʼּ������Ӽʼ��������H�;�/�����"�;�H�T�_�o�r�o�r�m�a�T�H������y�w�|�������������ݿѿĿ������0�)��$�.�<�I�U�b�w�u�q�o�j�b�]�U�I�<�0�������׾ʾʾʾо׾����������������H�G�C�D�H�P�U�Y�Y�U�H�H�H�H�H�H�H�H�H�H����ŶŶŲŬŭŹ���������������������������������������������$�%�����������)�)��)�/�6�8�B�F�B�6�5�)�)�)�)�)�)�)�)����޾۾޾������'�/�0�.�(�"��	��m�`�T�M�E�@�?�D�G�T�`�m�y�����������y�m�	�����ݾ׾Ѿվ׾����	���$�%�"��	�r�m�g�r�������������x�r�r�r�r�r�r�r�r�f�^�]�d�f�s�u�|�u�s�f�f�f�f�f�f�f�f�f�f�
��
����#�&�/�5�/�#�!��
�
�
�
�
�
������������ľ����������������
������ݿٿѿο˿ѿݿ߿�����������������������	�������	����¤¦²¿������������¾¶²¦Ňł�{�q�n�l�n�{ŇŔŚşŗŔŇŇŇŇŇŇ�_�S�H�@�A�S�l�r������������������x�l�_�6�,�)�(�)�)�+�6�B�O�[�\�h�h�c�[�S�O�B�6ÇÄÅÎÓàìù����������������ìàÓÇ�����r�N�@�3�+�2�s�����������������������T�N�H�B�;�9�;�H�T�a�m�n�z�}�|�x�p�m�a�T�4�/�(�0�4�9�A�J�M�P�R�M�C�A�4�4�4�4�4�4�ֺɺ��������ɺֺ����	��(�2�/����ֺɺź��źɺֺ�������ֺɺɺɺɺɺɿ�	�����׾˾Ӿ��	��.�D�R�G�A�.�"��̹����|�{�����Ĺܹ���'�@�`�b�Z�3���̿ݿӿѿĿ��ÿĿѿ׿ݿ��ݿݿݿݿݿݿݿݼ��������������������ܻۻлѻܻ����������������ܻܽнνĽ����ýĽнݽ�����������ݽ��V�V�P�U�V�a�b�o�s�o�o�e�b�V�V�V�V�V�V�V�Y�N�M�B�M�Y�Z�\�s���������������s�f�Y�M�K�A�@�@�A�M�Z�d�f�r�q�f�Z�M�M�M�M�M�M�Z�X�N�N�Q�Z�d�g�s�����������������s�g�ZƳƱƪƭƳƼ��������������������ƳƳƳƳ������������������$�0�4�7�3�0�-�$���"����#�,�8�;�H�T�\�a�]�W�M�H�<�3�/�"�z�m�i�m�z�����������������������������zŠŘŔŇńŇŔşŠšŭŵŹŹŹŭŠŠŠŠ�b�`�U�I�H�I�U�b�n�{�~�~�{�z�n�d�b�b�b�b�ù����������ùϹֹܹ�޹ܹϹùùùùùûû��������ûлܻ����������ܻлûû������л����4�M�]�\�U�@�'����л����������������������������������������޻l�d�_�[�_�a�l�q�x�y�y�x�l�l�l�l�l�l�l�l�B�>�B�F�M�O�[�h�tāčĎččā�}�t�_�O�B�3�1�2�3�?�@�L�Y�[�Y�U�L�F�@�3�3�3�3�3�3�L�K�H�F�H�L�Y�e�i�m�g�e�Z�Y�L�L�L�L�L�L���}��������������������������������������
����*�6�C�O�P�Q�O�K�C�C�6�*��"�"��!�"�/�4�;�H�I�Q�T�X�]�T�H�;�/�"�"������ĿĺĵĻĿ����������������������������������(�*�5�A�K�A�5�2�(��������������������
���������������������
�
���� �������������������������Ŀƿѿݿ���ݿܿԿѿʿĿ�ƎƉƎƖƚƧƳ��������ƳƧƚƎƎƎƎƎƎE�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E�E��@�4�+�'�$���#�'�4�@�E�I�K�M�M�M�M�A�@�g�a�b�e�g�s�w��������s�g�g�g�g�g�g�g�gE*E$E*E+E7ECEPE\E`EhE\EPECE7E*E*E*E*E*E*àßÓÓÊÇÅÆÃÇÓààåìóôòìàĦĚĚĔēĕĚĦĳĿ������������ĿĳĦĦ����������������������������������������a�]�U�M�R�U�a�n�r�w�n�d�a�a�a�a�a�a�a�a�h�`�[�Q�S�[�h�tāčĚěĚĔďčā�t�h�h�/�%�#���!�#�/�<�H�K�M�M�H�H�<�/�/�/�/�y�q�y�������������������������������y�y�������������ûлڻл˻û��������������� 9 % m 8 5 2 h g 7 H 3  + S k > [ & F . ^ ) W ( O O & 7 * & h [ H r ' " z B C  !  n u Q p , 5 ^ @ S I \ V u  /  ] d \ J e L ] A Z w  ~ $ L  � F}  �  1  �  @  P  E  �  \  �  	  D  �  �  �  ^  _  z  �  �  N     �  �  X  �  �  y  X  �  �  �  V  a  _  1  2  �  W  �  p  �  �  G  �  �  �  {    �  w  \  �  �  �  e  t  =  >    �  n    �  �  m  p  �  �  �  [  h  .  "  �  X  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  D�  g  #  �    �  �  �  	(  	K  	_  	Q  	6  	  �  q  �  �  �  ;  �  	�  	�  	�  
  	�  	�  	�  	�  	Q  	  �  {    �  �  I  �  %  v  �  '  #      �  �  �  �  �  �  �  �  �  �  }  a  L  6  !    }  �  �  �  �  �  �  �  �  {  Z  <    �  �  i    �  c  !  @  Z  s  �  �  �  �  �  x  T  !  �  �  �  n  1  �  5  t  �  N  a  \  Z  Q  7    �  �  �  }  ]  @    �  �  �  ]  �  �  0  F  [  W  E  1      �  �  �  �  �  �  i  O  5      �        	            �  �  �  �  �  �  �  �  �    $  i  a  X  N  =  +    �  �  �  �  �  \  3  	  �  �  e    �  �  �  �  �  }  e  G  (    �  �  �  V    �  n  �  �  D  �  r  b  R  B  1      �  �  �  �  �  �    g  N  ;  )      Z  �  �               �  �  s  3  �  �  L  �  �  )  Z  �  �  1  `  �  �  �  �  �  �  �  �  q  G    �  �  E  �  n       9  K  U  \  ^  [  S  D  ,    �  �  �  z  B  �  �  M  $    �  �  �  �  �  y  \  =    �  �  �  �  v  P  -      '  "          
    �  �  �  �  �  �  �  �  �  �  �  �          �  �  �  �  �  �  �  �  �  �  �  �  �  �  }  s  
2  
7  
0  
  
  	�  	�  	�  	F  �  �  F  �  O  �    \  <  �   �  W  G  7  %    �  �  �  �  �  [  4  	  �  �  �  ]  ;     �  �    C  �  �  �  �  �  8  Z  o  p  [  8  �  �    e  �  �    �  �  �  �  �  n  L  +    �  �  r  0  �  �  X  �  ]   �  �  �  �  �  �  �  �  �  n  W  >  %    �  �  �  �  �  �  �  /  K  V  W  H  �  {  i  R  /  �  �  Z  �  �  Q  �    m  l    M  m  �  �  �  �  q  S  0    �  �  B  �  \  �  :  �  !  U  �  �  �  �    +  C  R  T  K  *  �  �    �       6  �  �  �  �  �  v  M    �  �  �  L    �  �  :  �  {    �   ?    -  6  4  0  $      �  �  �  �  Q  *    �  �  >  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  h  W  J  >  3  '    �  �  �  �  �  �  �  U    �  �  J  �  �  "  �  )  �    G  F  2      �  �  �  p  A    �  �  W    �  �  b    �  d  �  t  j  n  v  �  �  �  z  Z  ;    �  �  �  h  *  �  ]   �  �  �  �  p  >  �  �  i  I  1    �  �    �    �  H  �    (  #            	       �  �  �  �  �  �  �  �  �  �        
    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  f  R  <     �  �  }  G    �  �  O  �  �  �  �  �  �  �  �  q  S  3    �  �  �  �    Z  0  �  �  �  �  �  �  �  u  b  L  4      �  �  �  b  0   �   �   �  "  #    
  �  �  �  �  �  �  �  �  q  U  #  �  v    �    �  �        �  �  �  �  �  �  �  i  Q  9  !    �  �  �             �  �  �  �  U    �  �  F  �  [  �  [  �   �  ,  %          �  �  �  �  �  �  �  |  [  :     �   �   �  �    1  =  C  @  5    �  �  �  q  /  �  �  -  �  ]  �  A    �  �  �  �  �  �  �  �  �  �  �  �  �  o  E    �  �  d  )  ;  &     �  �  �  �  �  �  �  �  �  �    �  �  �  �  �          �  �  �  �  �  �  �  |  b  I  -    �  �  n  '        �  �  �  �  �  s  U  8    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  Q    �  �  U  �  �  �  �  �  �  �  �  n  Q  1       �  �  U  �  \  �  �  �  l  S  9  .    
  �  �  �  v  �  �  �  �  {  4  �  j   �  D  9  >  ;  A  I  B  2  0  ,  )  /  6  9  9  2  !    �  �  �  �  �  �  �  �  �  w  d  Q  :      �  �  �  �  m  =    �  �  �  �  h  :    �  �  T  K  �  q  Q    �  Q  �    j  �  �  �  �  �  �  �  s  d  W  N  9    �  �  �  ]  %  �  �  ,  &         �  �  �  �  �  �  �  h  M  /    �  �  �  �  �  �  x  h  S  >  ,      �  �  �  �  �  �  �  �  ^  ,   �  �    $  .  -  $      �  �  �  �  f  8  �  �  a     W  9  �  �  �  �  �  �  �  t  M  &  �  �  �  H  �  d  �    d  �  z  �  �  �  �  �  t  a  M  9  %    �  �  �  f    �  `   �  �  �  �  �  o  M  )    �  �  }  N    �  �  �  H  �  �  z  _  U  L  B  6  (        �  �  �  �  �  a  A  $      �  C  7  *         �  �  �  �  �  �  �  m  G  !  �  �  �  ~  +  !        �  �  �  �  �  �  �  ~  a  >    �  �  �  /  p  b  S  E  8  *      �  �  �  �  �  �  t  [  A    �  A  2  O  f  z  �  �  �  �  k  6  �  �  h    �  �  .  �  �  I  �  )  H  Y  a  j  m  P  -    �  �  I  �  �  &  ^  b  z    '  "          	  �  �  �  �  �  �  �  �  t  b  O  =  *  z  t  l  c  X  J  9  '      �  �  �  �  �  _  	  �  '  `  d  l  c  Q  @  7  -      �  �  �  l  K  &    0  �  �  �  x  �  �  �  �  �  �  �  J  �  x  �  f  
�  	�  	  7  2    �      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  ~  }  �  �  �  n  P  0    �  �  �  �  \  1    �  �  f  '  �  �        �  �  �  �  �  ~  R     �  �  "  �  5  �  >  �  A  $  X    �  �  �  �  �  v  J    �  u    �    �    f  ;  �  �  �  �  �  �  �  �  �  z  i  R  :  #    �  �  �  �  m  �  {  ^  A  $    �  �  �  �  T  #  �  �  �  ^  +  �  �  �