CDF       
      obs    Q   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type         ����         	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�z�G�{     D  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       P[�     D  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       <D��     D   4   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F�G�z�     �  !x   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �׮z�H    max       @v|�����     �  .    effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @.         max       @Q`           �  :�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�>        max       @�e`         D  ;l   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       �%   max       ;o     D  <�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A�)�   max       B4��     D  =�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��   max       B4��     D  ?8   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >dB�   max       C�Ϋ     D  @|   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >@5*   max       C��     D  A�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          Q     D  C   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          5     D  DH   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          1     D  E�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       PO�     D  F�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��i�B��   max       ?�1&�x��     D  H   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       <D��     D  IX   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>��Q�   max       @F��G�{     �  J�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ��G�z�    max       @v|�����     �  WD   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @Q`           �  c�   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�>        max       @�{          D  d�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�     D  e�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�kP��{�   max       ?�/�{J#:     P  g                                                   7      .         
                              !   	                        	         	   	            P   J   	            	   B      	            <                                                      
   N���N͸N;#O��N�oKO��zO*��OO�N�o�O�p�Nc��O+�SOB�iOF�Oq�NEP[�O��bP�AN�~�O�YO�N�?N��%N���O-3Nղ[OxN#KN5�JN��O�y�N���N��N@R�N��YO_O~ �O2�,NOnOS�4M��NL�NڦiNM$�N�SEO�^�N��O�e�Oǹ8O	Q{O�sN�1�N�J�M��]P6ROt�4N�^�O*gN� RO$��O�4}O>N)OQI;N�=O,Z�O��#N�NҚ�O?��N���N��Onb1O?��O8�lN0O�i�N�OL�N��IN�W�<D��;��
;��
;D����o��o�o�o�t��T���e`B�e`B�e`B�e`B��o��C����㼛�㼣�
���
��1��1��9X�ě���h��h��h��h��h��h�������o�o�o�+�\)��P��P������w�#�
�,1�0 Ž8Q�<j�<j�<j�@��@��D���P�`�T���Y��]/�]/�aG��e`B�e`B�ixսixսm�h�m�h�y�#�y�#�y�#�}󶽃o�����7L��O߽�O߽�hs��hs��9X��vɽ�vɽ�`B�� $"�#+/34/+#DHMUY[abaUHBDDDDDDDD�������������������EO[[fhohh[WONJEEEEEE�������� ����������EHanuz~|unsurna[SSNE	+03<DILI<20#
	��������������������HMTamz������zmaTHEEH��� �������������#/<AHJIHDA</,)268=BOUY[bdc[YOB<512#0<DIMW\][UI<;%#��
#/<<;/#
��������������������{|�����	#<n������{^I0
	z~��������������{yyz )6B[t���thO6 ��������������������/;Haz�����zmaTH;/&'/$)5?=95));<HUYXYUYULH<:7;;;;;KOV[hkt�tmh[OHKKKKKK�������������������������

������������������������������!'!�������16BOQPOB601111111111�������������������������������9BGNUanz�����znUH<99��������������������./<HOMH<:/..........//5<HJKH<4/-////////^ht{�������tjhaa^^^^��������������������su���������������~xscgpt����������tjg[[c��������������������JNS[gt{�����ytg[OLIJz{}����{zuzzzzzzzzzz��������������������mt{�����������{wvtnm����������������������������������������jnz����������zmgehhj��������������������9Uanquxx~��na^UH<129���������������������������������������������������������������
���������������������������().67?6)&%((((((((((��� ���������#0=IJG<51//-"5<IPUVUUKI@<85555555������������������������������������������������������������/4<HUavopmjaUS<5/--/�����
������������
"/:/)
������AHIUanuyvnnaUUH>AAAA��������������������Ng�������������t[NHN��������������������;AGHJQRQNKHD=;54578;~�����������������~~������������������������


������������)**---)'	���(+/5BMNX[[UNKB=5-)((rtv}�����������wtrqr�������������~zx}��������������������25:BNNYNBB5122222222\gt|���������{tqg`[\�����������()35BNZTNB@5)%((((((���ֺӺӺֺ������������������z�t�k�g�a�g�t�{�����������������������������������������������������������������������	�����Y�M�Y�Y�e�g�r�~�~���~�s�r�e�Y�Y�Y�Y�Y�Y�T�M�F�D�E�N�Z�m�s�������������������T�H�;�9�4�/�$�/�;�H�a�m�z�������z�m�a�T�H���������������¼ʼ̼ϼʼȼʼͼҼʼ���������������������������	�
�	������������ƳƧƕƎƕƥƳƹ��������������������Ƴ�����������������������������������)�"�"�%�(�)�1�6�B�C�O�[�o�l�h�[�O�C�6�)�m�d�`�T�I�I�T�`�m�y�����������������y�m�û��������������ûлܻ������ܻۻл�àÝÞÖÔ×Ýàì����������������ùìà�����!�'�-�1�:�<�:�-�+�!������������l�\�B�8�N�Z�����������������������ݿѿ������ʿѿݿ�����"�(�4�(�"��������������2�=�C�H�V�`�h�i�g�a�T�H�/����"������"�/�/�;�=�>�;�/�"�"�"�"�"�"�����������
� �<�P�I�D�=�4� �������������¿¸º¿�����������������������������U�T�I�N�U�a�n�zÇÏÇÄ�|�z�n�a�U�U�U�U�ù������������ùƹϹҹڹعϹùùùùù����~�w���������������������������������������������������������������������������g�_�b�g�s�y�������������������s�g�g�g�g���������������������������������������������������������������������������������;�6�3�.�-�,�.�:�;�E�E�G�G�G�;�;�;�;�;�;���������	�����	���������������������s�g�Z�5����(�5�A�Z�g�t�������������s��������'�4�@�>�4�'���������������������������������������������������������
���"���
���������������������������������������������������������x�l�d�_�[�W�S�G�S�Z�_�l�x�������������x�������������������ʾ׾߾����ھ׾ʾ������߾������	�������	����׾ӾʾȾʾ׾����׾׾׾׾׾׾׾׾׾��)� ������%�)�/�5�B�[�e�d�[�N�B�5�)���������������������������������ìåçåìðù��������ùìììììììì�-�-�/�:�C�F�Q�S�_�l�x�����x�j�_�S�F�:�-F$FFFFFF$F,F1F<F7F2F1F%F$F$F$F$F$F$�@�7�@�A�L�Y�c�e�~�����������~�o�e�Y�L�@�ݿѿ������Ŀƿѿݿ����
���������ŹűŭŠśŔŏōŒŔŠŨŭŲŹ��������Ź�������������ܹ���3�=�1�����Թ������L�C�B�E�F�L�Y�r�����������������~�e�Y�L�6�4�*����
����*�6�:�C�J�K�C�C�6�6����������������������������ʾȾ������žʾ׾������������׾ʾ����	����*�6�C�N�C�?�6�*���������������������������������������������ҽ������������.�G�l������z�o�S�:�!������������������̽ݽ���������нĽ������&�(�1�4�A�C�A�?�4�(�������ܹٹչֹܹ߹��������������ܹܻ��������������ûлջٻлϻû��������������������	���'�4�7�4�'�����������Ҿ;ɾ׾���"�H�J�G�7�.�"��	���������������(�+�(�&� ������������������Ŀοѿӿݿ����ѿĿ������Ŀ������������Ŀѿݿ޿�ݿݿҿѿĿĿĿ��[�U�O�L�B�G�O�[�h�tāĉčďĆā�t�s�h�[�H�<�*�%�"�%�+�7�=�I�V�b�g�m�l�i�e�b�V�H����������������	���������������������K�A�5�3�5�A�N�Z�g�s�|�����������s�g�Z�KŔŇ�{�n�i�b�X�b�l�o�{ŔŠŭ����ŵŭŠŔ�f�_�e�r���������������������������s�fD�D�D�D�D�D�D�D�EEEEEEEED�D�D�D��	����������������"�/�9�;�B�A�;�5�"��	�����������������������������������������#���
�������
��#�/�H�U�Z�T�H�C�<�/�#¦¥£¦¨²¿������������������¿²¦¦������Ŀĺ����������$�+�-�#������������t�s�t�xāăčďččąā�t�t�t�t�t�t�t�tčČċčďĚĦĳĿ��������������ľĦĚč��������þ������������������������������àÞÓÒÏÐÓàâëìðòìàààààà < l B f a # m v = 4 C @ < C J f F D G M x 7 I K E # Y a h � C > P b H C j A ' 8 % O M v h v : h  4 @ O _ q V = o $ "  < B . ~ < ) g b � [ � ? . = U T 6 w G Q k    z  n  R  M  �  �  �  k  �  �  |  �  �  �    �  �  ,  �  �  B  K  �  �  �  R    d  O  �  ,    �  L  i  �  �    v  ,  �   �  t  4  �  O    U  �  �  2  o  �  �  (  �  :  �  C  �  l  �  �  �  )  m    N  }  �  *  �  �  �  �    ~  p  �  �  Ļo%   ;o�o��C��o�T����t��49X�,1��C��o��/���+��9X���P��w��+�����0 ż���h���'#�
�\)�+���+�o��o�#�
�t��t��,1�<j�y�#�aG���w�@��#�
�@��H�9�P�`�aG������Y��%��F�e`B�m�h�y�#�aG��u��F���-�}󶽗�P�}󶽙�����hs���T���P��-���w��%��\)���㽋C���9X��^5��vɽƧ𽟾w�������S����#�%BYRB�@B��B��BB"/gBBvB%T�B��A���B&aB&B��B&�B��B%?B'�B)��BK2B Q�A�j�BM�B]FBM_B8�BBcB3hB�=B+B-�OA�)�Bs�B iBԈB��BF�B!R�B4��B	�XB��B	9B(��BeB)$B�B"�B /hB��B��B�B-B�QBE!Bp%B�NB�mB%�yB&��B ��BR�B�7Bm>B]�B��B�B7~B
�gB
�PA�аBF3B+�zB'B�&B��B
�CB
�:B�;B�B	�hBP�B��BD(B�yB��B�B@�B"A|B��B$��B�A�w�B>�B�B�&B%�@B��BC�B'4�B)�B?`B I\A�ݝB}�B?PBA8B?qB?�B(&B�&BCaB.A��B@�B ?�B�
BǥB@B!E�B4��B	�aB�%B�iB(�DB�"B)@OB�^B"��B CB�B��B��B
�	B��B?�B>�BBB@1B%R$B&�LB �:BH�B��B@B&�B��BAEB6�B
��B
�hA��(BEzB+FaB>�B��B��B
��B
I�B��B��B
BE�B�@G��A��-A��rA���?�q�ACz�A�Q�@�[A�b�BF�A���Aؗ�Am�@�ҵA�,�@k�A��TA��A�1�A���A�MA�'�A�Ag>j;�A���A�>�A�w�A�ZA��SAa�tA�Z�A��*@�w_A�ɽA��S@��F@��AO?WAY�*AS�A���A1{A�/@���C�Ϋ?�OA}�A��>dB�@�A���A���AS��A�CiAб�Ao|A'��A7Q�?,��@���@A[y�A1? Az�:AyA+Aۅ~BD�B�A���A�ϋ@�L�C�N�A���A�vvA�s>A�ɒA斘A�H�A�$A�cA�|d@DMA��*A���A�j,?��ACq�A��@�)A�shBLA�xQA��[An��@�,nA�z�@qtYA���A=4A��A���A�oXA�k�Aƚ1>G��A��A�GA�f�A�#�A��Ac�A�]`A�v�@ƠA�}0A�$j@���@�'`AM�SAY�AT�"A�{5A0�#A�1�@��:C��?���A}W�A��>@5*@$�B @dA���ASaA�c�AЊA�OA$�bA7]�?/��@�U@��A[�A15nA{.�Ayk�A�`�Br0B��A�\OA�=�@�/,C�W:A�r�A�kA�[�A�;�A�� A݀WA�|A�
yAˆm                                                   8      /         
                              "   
                        
      	   
   
            Q   J   
            	   C      	            =                     	                           	                                       %                     5      )      %                                 %                                                   +                     )   !               #               !                                                                                             1            %                                                                                                                                          !                                          N���N͸N;#NM��N�oKOp�O�OO�N�o�O�#�Nc��O+�SOB�iO4O6�-NEPO�Ocy�O���N5�O�YN���N�?N4oN���O-3Nղ[OxN#KN5�JN��OD�N���N��N@R�N�wN֫�O,�N�V�NOnOS�4M��NL�NڦiNM$�N�SEOM@�N��O��vO��<O	Q{O�sN��vN�J�M��]O��2Oci�N�^�O*gN� RO	��O4_�O>N)Oo�N�-O,Z�O��#N�NҚ�O"x�N���N�=�OLI�O3h�N߱;N0O�)�N�OL�N��IN�W�    �  �  f  v  �  b  �  ]  �  �  C  �  �  }  	  �  �  6  �  �  /  R  F  �  �  �  {  �  �  H  �  U  }  �  z  �  �  S    �     �  ]  �  �  #  5  S  
u  i  �  k  9    b  �  l  l  B  �  �  ]  /     N  *  %  �  U    _  �    �  �  �  b  t  �  �<D��;��
;��
�o��o��o�t��o�t���t��e`B�e`B�e`B�u���㼋C���1��1�\)��1��1��9X��9X��`B��h��h��h��h��h��h���0 ż��o�o�C��\)�',1��P������w�#�
�,1�0 ŽH�9�<j��%�Y��@��@��H�9�P�`�T�������aG��]/�aG��e`B�m�h�����ixս}�q���y�#�y�#�y�#�}󶽅������C���hs��\)���w��hs��E���vɽ�vɽ�`B�� $"�#+/34/+#DHMUY[abaUHBDDDDDDDD��������������������EO[[fhohh[WONJEEEEEE��������������������GH`nurpnknrtpna\UUPG	+03<DILI<20#
	��������������������GHKTamz�����}zma]THG��� �������������#/<AHJIHDA</,)268=BOUY[bdc[YOB<512#0<AIKV[[UQI<'#���
#172/)#
������������������{|�����
#<Un{�����{\I0#

z|�������������}{zzz',6BDORcgia[OB6)&""'��������������������/;Haz�����zmaTH;/&'/ )58:55);<HUYXYUYULH<:7;;;;;NOZ[`hqhg[OLNNNNNNNN�������������������������

������������������������������!'!�������16BOQPOB601111111111�������������������������������UVanuz���zunaUQKNQU��������������������./<HOMH<:/..........//5<HJKH<4/-////////ghht�����tphccgggggg��������������������������������������|�`gjt��������tqga````��������������������JNS[gt{�����ytg[OLIJz{}����{zuzzzzzzzzzz��������������������mt{�����������{wvtnm����������������������������������������mqz�����������zmllkm��������������������8<HUainqrrpkbULH>968��������������������������������������������������������������� 
 �������������������������().67?6)&%((((((((((������	����������#0<GHF<:40-#5<IPUVUUKI@<85555555������������������������������������������������������������<HUV_acdca[UHA<8547<�����
����������
#)##
�������BHKUantxunlaXUIHBBBB��������������������Ng�������������t[NHN��������������������;AGHJQRQNKHD=;54578;�������������������������������������������

	��������������&)+,,)%���)+05BKNWZZTNJB;5/+))ttv|���������}uttttt�������������~zx}��������������������25:BNNYNBB5122222222\gt|���������{tqg`[\�����������()35BNZTNB@5)%((((((���ֺӺӺֺ������������������z�t�k�g�a�g�t�{��������������������������������������������������������������������������������Y�M�Y�Y�e�g�r�~�~���~�s�r�e�Y�Y�Y�Y�Y�Y�s�n�f�Z�V�O�V�Z�f�s���������������s�s�H�C�;�9�;�H�T�[�a�d�m�z�������z�m�a�T�H���������������¼ʼ̼ϼʼȼʼͼҼʼ���������������������������	�
�	����������������ƳƜƓƚƬƳ������������� ��������������������������������������������)�"�"�%�(�)�1�6�B�C�O�[�o�l�h�[�O�C�6�)�m�d�`�T�I�I�T�`�m�y�����������������y�m�û����������������ûлܻ����ܻڻл�ìäââàÙ×àìùþ��������������ùì�����!�'�-�1�:�<�:�-�+�!�����������n�^�P�E�D�I�Z������������������������ݿѿ¿��ĿϿݿ�������&��������"��	��	���/�;�H�T�X�^�_�_�Z�T�H�;�"�"� �����"�-�/�;�/�,�"�"�"�"�"�"�"�"�����������
� �<�P�I�D�=�4� �������������¿º¼¿�����������������������������U�T�I�N�U�a�n�zÇÏÇÄ�|�z�n�a�U�U�U�U�ù����������ùιϹֹҹϹùùùùùùù����~�w���������������������������������������������������������������������������g�_�b�g�s�y�������������������s�g�g�g�g���������������������������������������������������������������������������������;�6�3�.�-�,�.�:�;�E�E�G�G�G�;�;�;�;�;�;���������	�����	���������������������7�5�(�)�5�:�A�N�Z�g�n�s�x�y�u�s�g�Z�N�7��������'�4�@�>�4�'���������������������������������������������������������
���"���
���������������������������������������������������������x�w�l�h�_�]�[�W�_�l�w�x������������x�x�������������������ʾϾ׾پݾܾؾ׾˾��������������	�������	���������׾ӾʾȾʾ׾����׾׾׾׾׾׾׾׾׾��)� ������%�)�/�5�B�[�e�d�[�N�B�5�)���������������������������������ìåçåìðù��������ùìììììììì�-�-�/�:�C�F�Q�S�_�l�x�����x�j�_�S�F�:�-F$FFFFFF$F,F1F<F7F2F1F%F$F$F$F$F$F$�@�7�@�A�L�Y�c�e�~�����������~�o�e�Y�L�@�ݿѿ¿����ĿͿѿݿ��� ����������ŹűŭŠśŔŏōŒŔŠŨŭŲŹ��������Ź�����������������Ϲܹ��������Ϲù����e�Y�L�J�K�P�Y�r���������������������~�e�6�4�*����
����*�6�:�C�J�K�C�C�6�6����������������������������׾;ʾ������ɾʾ׾׾�����������׾����	����*�6�C�N�C�?�6�*���������������������������������������������ҽ.�!�������!�.�G�V�c�h�i�`�S�G�:�.�������������������ɽнݽ������нĽ������&�(�1�4�A�C�A�?�4�(�������ܹٹչֹܹ߹��������������ܹܻ��������������ûлջٻлϻû������������������������'�3�4�5�4�'����������������	���.�7�3�.�)�"��	����������������(�+�(�&� ������Ŀ��������Ŀ˿ѿݿ޿�����ݿ׿ѿĿĿĿ������������Ŀѿݿݿ�ݿܿѿпĿĿĿ��[�U�O�L�B�G�O�[�h�tāĉčďĆā�t�s�h�[�H�<�*�%�"�%�+�7�=�I�V�b�g�m�l�i�e�b�V�H����������������	���������������������K�A�5�3�5�A�N�Z�g�s�|�����������s�g�Z�KŔŇ��n�m�s�{ŇŔŠŭŹſſŹųŭťŠŔ�f�_�e�r���������������������������s�fD�D�D�D�D�D�EEEEEEEED�D�D�D�D�D��	������������"�/�5�;�A�?�;�3�/�"��	�����������������������������������������/�.�#�����#�/�<�H�R�M�H�<�1�/�/�/�/¦¥£¦¨²¿������������������¿²¦¦������Ŀĺ����������#�*�+�#������������t�s�t�xāăčďččąā�t�t�t�t�t�t�t�tčČċčďĚĦĳĿ��������������ľĦĚč��������þ������������������������������àÞÓÒÏÐÓàâëìðòìàààààà < l B q a  p v = 4 C @ < @ T f E ; ! @ x 9 I G E # Y a h � C % P b H 6 [ = + 8 % O M v h v / h > + @ O ] q V 1 n $ "  8 0 . K 5 ) g b � < � B - 9 H T 6 w G Q k    z  n  R  �  �  K  <  k  �  \  |  �  �  �  �  �  �  �    Q  B  !  �  P  �  R    d  O  �  ,  �  �  L  i  �    |    ,  �   �  t  4  �  O  �  U    m  2  o  �  �  (  R  �  �  C  �  1    �  L    m    N  }  q  *  �  �  �  �    i  p  �  �  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�          �  �  �  �  �  �  �  �  l  Q  6      �  �    �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  |  w  s  n  i  d  _  8  A  J  R  S  O  O  P  ]  ]  F  +           �  �  3  �  v  G    �  �  �  r  >  1  (  �  �  �  O    �  �  d  ,  �  %  C  X  e  q  {  �  �  �  �  z  i  U  :    �  �    p   �  ^  _  `  a  `  T  H  <  0  &        �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  s  `  L  7  !        �  �  �  ]  [  Y  W  U  S  Q  O  N  L  I  E  A  =  9  5  1  -  (  $    i  �  �  �  i  J  (    �  �  �  b  $  �  �    �  �  ]  �  �  �  �  �  �  ~  x  s  m  h  c  ]  X  R  L  F  @  ;  5  C  6  %      �  �  �  �  �  �  q  E    �  �  x  3  �  �  �  �  �  �  v  f  V  H  >  5  *      �  �  �  |  >   �   �  �  �  �  �  �  �  �  �  v  S  .    �  �  M    �  �  �  �  O  h  v  {  z  t  j  Z  D  "  
  �  �  �  �  �  �  �  �  �  	  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z  e  O  �  �  �  �  �  ~  i  M  +    �  �  W    �  k  �  �    �  c  �  �  �  w  c  J  ,    �  �  �  �  p  U    �  �  P  s  J  �  �  �    (  5  3  +      �  �  �  L    �    [  �  �  �  �  �  �  {  h  N  3    �  �  �  �  }  [  5   �   �   �  �  {  c  E  !  �  �  �  g  q  ^  4  
  �  �  :  �  �  D  R     '  .  )  #      �  �  �  �  �  �  l  O  2    �  �  �  R  G  <  1  &      
    �  �  �  �  �  \  0     �   �   �  �  �    6  A  E  C  =  2     	  �  �  �  �  �  h  0  �  f  �  �  }  u  s  t  y  y  u  e  Q  7    �  �  �  �  �  c  B  �  �  �  �  �  �  �  �  �  �  �  t  `  H  5  6  !  �  �  �  �  �  �  �  �  �  �  z  n  b  T  D  4        �  �  �  p  H  {  z  x  v  t  q  j  b  Z  S  J  @  5  +          �   �   �  �  �  �  �  �  �  �    {  x  s  m  f  _  Y  R  K  E  >  8  �  s  ^  I  4      �  �  �  �  v  W  9     �   �   �   �   �  H  C  >  9  4  /  *  %  !        	    �  �  �  �  �  �  e    �  �  �  �  �  �  �  �    Z  ;    �  �  Z  �  S  c  U  O  I  J  K  G  A  >  <  :  6  2  +  $        �  �  �  }  x  s  m  h  b  W  L  @  5  (      �  �  �  �  s  J  "  �  �  �  �  �  �  �  �  �  �  �  �  �  y  n  `  Q  B  3  $  d  m  v  x  y  u  l  Z  F  0    �  �  �  �  w  Q  +    �  �  �  �  �  �  �  �  a  >      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  <  �  �  x  $  �  1  <  E  M  Q  S  P  K  @  1    �  �  �  N    �  �  =  ~          �  �  �  �  �  �  �  �  �  �  �  �  {  k  \  M  �  �  �  �  �  �  �  �  q  X  :    �  �  �  g  ;     �   �     �  �  �  �  �  �  �  �  �  �  �  �  �  z  q  h  _  V  M  �  �  �  �  �  |  s  i  T  =    �  �  �  �  \  2  �  �  r  ]  A  %    �  �  �  �  �  �  �  �  {  X  7    �  �  H   �  �  �  �  �  �  �  q  P  -  �  �  �  I    �  �  `  #   �   �  �  �  �  �  �  �  i  9  	           �  �  �  �  e  H  +         #      �  �  �  �  g  :    �  �  8  �    ]  �  5  4  3  /  "      �  �  �  �  �  �  �  �  �  �  Z  %  �  
|  8  �  <  R  =    �  �  ,  
�  
G  	�  	@  �  �  �       �  
8  
K  
u  
g  
T  
:  
  	�  	�  	�  	^  	  �  G  �    "  �  A  �  i  a  Y  R  J  @  6  ,  $        �  �  �  �  �  x  {    �  �  �  �  z  f  R  >  .  !        �  �  �  {  D  $    i  j  b  P  ,    �  �  �  a  3  �  �  �  ;  �  �  \     �  9  ,        �  �  �  �  �  �  �  �  �  �  �  �  �    	          
    �  �  �  �  �  �  �  �        �  �  �  F  �  �    =  W  a  ]  B    �  v    �    �  �      �  �  �  �  �  t  U  =  =  G  E  <  '    �  �  �  x  4  �  0  l  a  V  I  <  -      �  �  �  �  �  �  n  <    �  w  *  l  S  Y  >  :  ,    	  �  �  �  �  �  �  S  #  �  �  �  V  B  <  6  1  &        �  �  �  �  �  �  �  �  �  �  �  s  �  �  �  �  �  �  �  �  b  =    �  �  �  �  �  c  ;  7  =    e  �  �  �  �  �  �  �  �  z  A  �  �  !  v  �  �  �  D  ]  X  O  B  1      �  �  �  �  ]  6    �  �  c  4    !  �  �    #      �  �  �  �  t  D    �  _    �  x  C  #  �          �  �  �  �  �  g  <    �  �  `    �  �  �  N  5    �  �  �  �  r  J    �  �  �  ;  �  �  ?  �  V  �  *      �  �  �  �  e  <    �  �  �  �  b  5     �  o    %  %  %  &  &  &  '  '  '  (  (  '  '  '  '  &  &  &  &  %  �  �  w  X  8    �  �  �  �  �  �  �  �  {  X  3    �  �  <  L  L  6     
  �  �  �  �  �  �  �  u  e  U  E  5  4  ?    �  �  �  �  �  �  �  �  �  �  �  y  h  V  C  0       �  O  \  7    �  �  u  4  �  �  ^    �  �  9  �  �  O  �  u  �  �  �  s  P  *    �  �  �  b  7    �  �  y  F  �  �          �  �  �  �  �  x  X  3  
  �  �  �  T    �  �  M  �  �  �  �  �  �  �  �  �  �  w  M    �  �  u  &  �  �     �  y  p  e  U  E  -    �  �  �  �  �  ~  y  ~  �  ~  u  l    ~  t  h  [  K  5    �  �  �  �  Q    �  S    �  B  �  b  O  ;  ?  P  Q  !  �  �  y  8  �  �  b    �  ~  0  �  �  t  m  b  R  S  >    �  �  �  y  A    �  x  ,  �  �  0  �  �  �  �  �  q  N  +    �  �  �  �  T    �  �  S    �  u  �  �  �  �  �  �  k  N  ,     �  }  7  �  �  P  �  �    c