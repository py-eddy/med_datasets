CDF       
      obs    >   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?h�t�j~�   max       ?�I�^5@      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       N��   max       P��`      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       �\)   max       =ě�      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F�����     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       �ۅ�Q�    max       @vvz�G�     	�  *D   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @,         max       @P            |  3�   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @�r        max       @�+@          �  4p   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��   max       >��;      �  5h   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A��Y   max       B0)�      �  6`   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A��?   max       B0>�      �  7X   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       >�HJ   max       C�x�      �  8P   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       >���   max       C�lb      �  9H   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max               �  :@   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min          
   max          ;      �  ;8   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min          
   max          3      �  <0   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       N��   max       Pg��      �  =(   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�[W>�6z   max       ?� hۋ�q      �  >    speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       �\)   max       >&�y      �  ?   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>�G�z�   max       @F�����     	�  @   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ���Q�    max       @vvz�G�     	�  I�   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @#         max       @P            |  Sp   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @�r        max       @�x`          �  S�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         Am   max         Am      �  T�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?�Ov_خ   max       ?� hۋ�q     `  U�      "      	                 %                  1            `   +            =            *   
   Z      ]   Z         +   6            6                   N   ;      3            '   	   
   	            N��O�N1��O��P��`O�O>�uNp�OFU�P�~N��JN�m�O��N�l{O	��P~�PO�n�O�B�NFt�Pb) N�(�O�
{O,lO � O�<NO$�O�N1�O�U�O��O��[NG&�O�[P8f�OȣtOc��OncP��dO��2Ow� O�LmP�+O�J�N�AO��N#/
N\ĉP��OxD�O+��OՋ�OG�O3�GN%#HO�N,�N�8N��O��N���N<=�N�a��\)��󶼬1�o��o%   :�o:�o;o<t�<t�<#�
<#�
<49X<e`B<u<�C�<�t�<�t�<���<���<���<�1<�9X<�9X<�j<�j<ě�<���<���<�/<�/<�=+=C�=C�=C�=\)=�P=��=��=,1=0 �=0 �=8Q�=@�=Y�=]/=aG�=q��=}�=}�=�%=�7L=�C�=�\)=��=��P=��w=�^5=��=ě���������������������77BCDDO[t����~tg[NG7��������������������qrr{��������������{q)5t�������wg[5//45?BN[\`ba[PNB?5//��)5BC?95,)�MNOV[_hljtxtha[OMMMMGCBNgt�������wtg[PNGwt�����������������w/*+-00<AIMMPJI<0////������	���������rrw����������������r����
!
���������")/;HMHDE;/'"�����"5EMB8)����� ,6AOU`fmomh[OB;3.wz�����������������w��������������������������)>EG@5)�����������


������T`nppz���������zn^VT���������������������}��������������������������������������F@@BGHT\admmromaTSIF����"(&���UH</-/;<HU[UUUUUUUUU'$'/<HUanpkuniaUH</'aWX\]__anz�����|znca���
#/9>AA>6/#
��}�����������}}}}}}}}�������')1)������������
1!3<?;/#
������������������������������
#)--*#	�����������

�������
$;BN[db[NIFA5)����������������������������������������������������������������
#3GU[VH#
���
	
)+6DAAA@6)""##%0790'#����������������������������������������"-/;<;/""�������$*10)�����������
 '--)#
���LKN[grt}����tng[UPNLMMVTU[ht��������t[OM����������������#&/<HPJH</#(!(*36CCDC6*((((((((ehlrt{���������thhee��������������������TZagmqzz{zxmaZTTTTTT)+26BO[[[ZSOB6)||�����������������|�����

���;9;HJTUWTHB;;;;;;;;;z~����������������zz�0�=�?�?�=�5�0�,�&�.�0�0�0�0�0�0�0�0�0�0�<�H�a�nÓàìöðëàÓ�z�n�N�:�1�,�/�<����������������������������������Y�f�r�v�����������r�f�Z�Y�M�H�E�M�Q�Y�)�[čĥĦğēā�h�O�6�������������)�n�{ŇŉŔŖŗŕŔŇ�{�n�f�b�`�`�b�f�n�n�G�T�`�g�m�p�w�w�m�b�`�T�G�B�;�9�:�?�C�G�y�������������������������y�w�v�y�y�y�y�����������������������������������������(�5�C�N�S�T�O�A�5������߿ѿ����#�(�������������������������~�w�~�����H�T�W�a�e�a�Y�T�H�;�6�6�;�D�H�H�H�H�H�H�ݽ������(�1�'����ݽĽ��������Ƚ���������������������¿�������������������H�T�a�c�m�t�z�����{�z�v�m�a�W�M�H�E�G�H�����O�K�5�(����ݿ��������������Ŀݾ����׾����ʾ�����s�d�V�]�l��������M�Z�]�f�s���������������f�Q�K�L�G�G�M�������ʼӼ˼ʼ�����������������������������<�`�b�[�[�L�0���������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D|D}D�D�D�D������#�/�4�/�����
���������������������ûлػۻػлʻû�������������������E�E�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E͹����ùϹ�����,�'����Ϲù���������������������	������ݿѿпϿпѿݿ�.�;�G�O�T�`�`�T�G�;�.�"������"�'�.²®­ª²·¿¿����¿²²²²²²²²²�������!�!��������������������������(�5�A�E�A�:�5�)�(���� ������D�EEE'E*E*E"EED�D�D�D�D�D�D�D�D�D�D�ÇÌÐÇÃ�z�n�n�k�n�z�}ÇÇÇÇÇÇÇÇ�����ʼ�����ּʼ��������������������B�[�g¨¡�g�N�5�)� �����%�B�T�m�y�����������y�`�T�G�D�E�;�7�6�;�G�T�	��"�.�;�@�8�.�*�"��	����۾׾����	�ݽ����(�A�D�4�&������ݽҽ̽нս�Ƴ����0�I�`�V�;�������ƧƁ�u�j�f�kƎƳ�����������4�*�����������Źūšŭ�������������������������������������������N�Z�_�g�s�����������s�Z�N�A�5�9�6�9�A�N����"�;�H�_�j�h�T�"�	��������������������!�-�:�S�_�k�c�_�Q�:�-�!������������������������������}���������	��"�.�6�;�A�;�:�.�*�"��	��������	�	�����z�m�h�c�m�z�������������������������
��#�#�&�#�!�����
�
��
�
�
�
�
�
�~����������ݺɺ����~�r�Y�L�H�K�T�p�~EiEuE�E�E�E�E�E�E�E�E�E�EuEiEdE`E^E`EeEi��"�&�.�1�/�.�"�����������������	����'�M�f�r���������r�Y�@�4������������`�l�y�������������������y�x�l�b�`�\�X�`����������������������������Žżŵż�ƿy���������������y�x�t�x�y�y�y�y�y�y�y�y�Y�^�e�r�y�~���~�z�r�e�Y�L�J�C�D�L�L�Y�Y�H�U�a�k�a�U�U�H�<�3�<�D�H�H�H�H�H�H�H�HŭŹ��������ŻŹŭŤŠŘŚŠŭŭŭŭŭŭ����
��������
������������������������������������ùìÓ�z�~Éâìôù���C�O�\�h�p�u�{�u�h�\�Z�O�H�C�6�4�6�B�C�C�P�\�^�]�\�Q�O�C�:�9�C�P�P�P�P�P�P�P�P�P�4�@�M�M�T�Y�\�Z�Y�M�H�@�>�6�4�3�0�.�4�4 ] T ` . $  . Y l C ( / ; 7 S 4 P D @ ! ; M 0 I i O + � 0 <  T b ! C H L ` k U D L > Y * B ^ V # % ` @ H P , z  g i R V _  B  E  o  3  j  ,  �  �  2  �  �  �  �  �  [  �    �  k  �    J  ;  ,  8  �  \  �     a  �  �  �    �  �    �  �  K  Y  �  D  X  T  C  �  �  �  l  ;  F  �  E  %  U  �  8  �  �  M  0��;D����C�%   >��;<u<49X;�`B<�1=8Q�<u<��
<���<���<�`B=�%=#�
=D��<�9X=�x�=y�#=��=��=<j=���<��=C�=C�=�+=\)=�h=�P=��m=��#=ix�=P�`=��-=�9X=�o=q��=}�=\=���=<j=Y�=Y�=ix�>$�=�l�=��-=�`B=���=��T=�\)=��=��w=��=��=��=�"�=���=�/B#PB��B�RB)w�B��B
�BZ�Bp�B	S�BC�B&:B�B֖B�eA��B�!B�0B8�B"v�B��B"Bq�B!��B0xB�?A��MB"�BaB�B�RBthB�B�ABQ�B�gB��B#~�B>�B�B��B�]B�'B�"B%i�B"}zB ��A��YBvLB��B	KGB��B-`B#B0)�B��BA�zBGB�uB�6A���B��B@�B	@�B�;B)�B��B@PB��B?*B	;�B�?B&"�B��B�rB�A�}XB�BA�BA$B"S�B>�B8BDAB!�(B@]B8"A���B�qB=PB�wB�ZB��B;=B�vB>�BIBB�B#��B�cB>�B��B��BBF�B%��B"?)B ��A��?B=�B�SB	<cBOB- �B�7B0>�B��B�`A��Bi�B@/BοA���B�MB
gtA��AA�+~@�?AٕQA�SAgx�A\A�SA�l�@�p�A���A-��A�PaA���A���AKACn�@��,A�C�ӝA��@���C�x�>�HJA�:�Ab^�A��Aұ8A���C�HTA�҇@���A�_�AiU�A[��A0��B�7A�f�A���A�|�A���@vT�AG��A]�#A�,�A�>�@�7C��!A�
{@�z,A8A�D�An��?ܵ�A�/�A��2A�G�A�i7B��B%l@��B
@�AȜ�A��@���Aق�A�Af�SAqA��A���@�cA�W�A-H'A��<A���A��cAL��AD @�.�A��C�ՏA��@�$�C�lb>���A��Ab*�A���AҁTA�i�C�G�A�h�@��A��Ag�A[}A0��B»A��?A�|�A���A��@w�wAG!A^�~A�lA�u@aIC��tA���@�Z�A�nA�� Am��?�q�AŊ�A��vA�]�A��B��B u@�\h      #      	                 %                  1            `   +            >   	         *   
   Z      ^   Z         ,   6            6   !      	         N   ;      3            '   	   
   	                  #         ;               '         !         5   '         /               %                        '   /            ;            +            
      +         #                                       !                        !                     #         %                                          !            3            !            
                                                N��O�X�N1��N�~6O���O�O%�iNp�N��Oʭ(N��JN�m�O�SN�l{O	��O�V�O���OG�/NFt�P�MN�ȠOx�O,lO � O���O$�O�yN1�N�c�O��OJ�=NG&�Oy��O��O��Oc��O#Pg��OC�YO$��Oz�cO�sO�J�N�AO��N#/
N\ĉO�\ O�/O+��O�K�OG�O3�GN%#HN�KON,�N�8N��O��N���N<=�N�a�  w  �  Y  �  �  8  �     �  �  9  �    ]    d  �  �  �  	�  8  �  R  �  	~  �  �  �  �    �  n  �  �  �    �  M  2  ~  �  �  �  S  n    �  	  	�  Y  x  �  �  �  �  ~  �  �  �  O  y  ��\)������1��`B>&�y%   ;o:�o<t�<e`B<t�<#�
<�o<49X<e`B=,1<��
<�`B<�t�='�<�j<��
<�1<�9X=C�<�j<ě�<ě�=49X<���=e`B<�/=q��=�%=��=C�=<j=�w=0 �=0 �=#�
=]/=0 �=0 �=8Q�=@�=Y�=���=��=q��=���=}�=�%=�7L=��-=�\)=��=��P=��w=�^5=��=ě���������������������>=EGHKW[t����|tmg[N>��������������������svz���������������{s<::?BIN[gt}{tg[ND<//45?BN[\`ba[PNB?5//  )3A=85*)$ MNOV[_hljtxtha[OMMMMVQT[gst���{tg[VVVVVV��������������������/*+-00<AIMMPJI<0////������	���������������������������������
!
���������")/;HMHDE;/'"�������	����!$06BO]bjljh[OJB>50!���������������������������������������������)5>A=5)�����������


	������aXVanqrz��������zna���������������������}��������������������������������������F@@BGHT\admmromaTSIF���� &$���UH</-/;<HU[UUUUUUUUU//1:<HHUVZURH<3/////aWX\]__anz�����|znca � 
 #/5894/#
}�����������}}}}}}}}��������������������
"&(%
��������������������������������
#)--*#	�����������

���������)BN[a`YKFC5)�������������������������������������������������������������
#.=AHKG</#

	
)+6DAAA@6)""##%0790'#����������������������������������������"-/;<;/""�����������������
 #%##
���LKN[grt}����tng[UPNLUSU\\^ht��������|t[U����������������#&/<HPJH</#(!(*36CCDC6*((((((((niptz��������tnnnnnn��������������������TZagmqzz{zxmaZTTTTTT)+26BO[[[ZSOB6)||�����������������|�����

���;9;HJTUWTHB;;;;;;;;;z~����������������zz�0�=�?�?�=�5�0�,�&�.�0�0�0�0�0�0�0�0�0�0�<�H�aÇÓàìðëäàÓ�z�n�U�H�?�6�0�<����������������������������������Y�f�r�}�����������r�f�^�Y�M�J�H�M�S�Y�B�O�[�h�m�t�x�|�{�t�h�[�O�6�&�!�%�,�6�B�n�{ŇŉŔŖŗŕŔŇ�{�n�f�b�`�`�b�f�n�n�G�T�`�b�l�m�r�u�m�`�T�G�D�;�:�;�<�A�E�G�y�������������������������y�w�v�y�y�y�y�����������������������������������������(�5�G�L�N�M�J�@�3�(������������(�������������������������~�w�~�����H�T�W�a�e�a�Y�T�H�;�6�6�;�D�H�H�H�H�H�H�ݽ�������������ݽнʽннԽݽ���������������������¿�������������������H�T�a�c�m�t�z�����{�z�v�m�a�W�M�H�E�G�H�ݿ�����#�$���������ݿ׿Կӿ׿ݾ����׾޾��ʾ�����s�l�b�b�f�s��������f�s�����������������s�f�b�Z�Y�X�Z�Z�f�������ʼӼ˼ʼ�����������������������������0�=�G�I�I�F�=�0�������������������D�D�D�D�D�D�D�D�D�D�D�D�D�D�D~DD�D�D�D����������#�/�3�/�#���
�������������仪���ûлػۻػлʻû�������������������E�E�E�E�E�E�FFFE�E�E�E�E�E�E�E�E�E�E͹������Ϲܹ��������Ϲù���������������������	������ݿѿпϿпѿݿ�.�;�G�L�Q�T�_�T�G�;�.�"������"�*�.²®­ª²·¿¿����¿²²²²²²²²²���������������������������������������(�5�A�E�A�:�5�)�(���� ������D�D�D�EEEEE EEEED�D�D�D�D�D�D�D�ÇÌÐÇÃ�z�n�n�k�n�z�}ÇÇÇÇÇÇÇÇ���ʼּ߼�����ּʼ������������������g�t�v�g�[�N�B�5�.�'�&�-�<�N�[�g�T�`�y�����������y�m�`�T�M�J�C�:�9�?�G�T�	��"�.�;�@�8�.�*�"��	����۾׾����	��������(�/�(�������ݽ۽ֽݽ��Ƨ��������0�I�G�1�������ƧƁ�m�h�nƎƧ���������������������ŹŲůŽ�����������������������������������������������A�N�Z�\�g�s�����������o�Z�A�9�8�:�8�;�A���	�"�;�H�U�b�`�T�/�"��	����������������!�-�:�S�_�k�c�_�Q�:�-�!������������������������������}���������	��"�.�6�;�A�;�:�.�*�"��	��������	�	�����z�m�h�c�m�z�������������������������
��#�#�&�#�!�����
�
��
�
�
�
�
�
�������ɺҺҺú������~�r�e�`�`�i�r�~����EuE�E�E�E�E�E�E�E�E�E�EvEuEiEgEfEiEnEuEu��"�&�.�1�/�.�"�����������������	����'�4�M�f�r�����x�r�f�Y�@�4�'�!�����`�l�y�������������������y�x�l�b�`�\�X�`����������������������������Žżŵż�ƿy���������������y�x�t�x�y�y�y�y�y�y�y�y�L�Y�e�o�r�w�r�p�e�Y�O�L�G�H�L�L�L�L�L�L�H�U�a�k�a�U�U�H�<�3�<�D�H�H�H�H�H�H�H�HŭŹ��������ŻŹŭŤŠŘŚŠŭŭŭŭŭŭ����
��������
������������������������������������ùìÓ�z�~Éâìôù���C�O�\�h�p�u�{�u�h�\�Z�O�H�C�6�4�6�B�C�C�P�\�^�]�\�Q�O�C�:�9�C�P�P�P�P�P�P�P�P�P�4�@�M�M�T�Y�\�Z�Y�M�H�@�>�6�4�3�0�.�4�4 ] W ` 2   1 Y C < ( / , 7 S   R / @  9 G 0 I _ O + �  <  T E  5 H > \ O T F 9 > Y * B ^ @  % \ @ H P . z  g i R V _  B  �  o    +  ,  e  �  �  �  �  �  =  �  [  �  �  �  k  |  �    ;  ,  q  �  3  �  �  a  �  �    |  3  �  L  ;  �  �    �  D  X  T  C  �  ]    l  .  F  �  E  �  U  �  8  �  �  M  0  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  Am  w  v  u  s  q  n  k  h  [  A  &    �  �  �  �  o  L  )    �  �  �  �  �  �  �  c  =    �  �  q  0  �  z  �  �    �  Y  S  M  G  A  :  2  )  !      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  u  e  T  B  /     �   �  
[  �  0  s  �  �  �  �  w  ,  �  �  �  o  �  �    �  8  X  8  +      �  �  �  �  �  �  �  �  p  ;  �  �  Q  �  �  -  �  �  �  �  �  �  �  �  �  �  �  �    k  R  7    �  �  y     
      2  G  [  ^  V  O  E  9  -  !      �  �  �  �  }  �  �  �  �  �  �  �  �  �  �  �  �  �  \  (  �  �  �  Q  d  v  �  �  �  {  l  T  5  6  H  F  8        �  g  �  >  �  9  +      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  {  `  ?    �  �  �  �  �  q  a  �  �  �  �  �     	      	      	  
    �  �  �  3   �  ]  Z  V  Q  M  G  F  F  A  6  %      �  �  �  �  �  |  T        �  �  �  �  �  �  �  |  O    �  �  �  Y  -    �  R  _  Y  O  Y  y  �  �  +  Q  `  c  X  >    �    E  T   �  y  �  �  �  �  �  ~  m  Y  C  *    �  �  �  t  H    �  $  W  �  �  �  �  �  �  �  �  �  �  �  a  .  �  �  1  �  8  �  �  �  �  �  �  �  �  x  m  b  Z  U  P  L  G  (     �  �  �  �  	W  	�  	�  	�  	�  	�  	�  	`  	8  	  �  �  /  �  O  �    �  �    .  4      �  �  �  �  n  M  )  	  
�  
�  
�  
k  
?  	�  N  �  �  �  �  �  �  �  z  �  �  �  r  a  K  )    �    �  _  R  P  J  D  >  <  ;  7  /  !    �  �  �  �  d  E  1    �  �  �  �  b  G  1       �  �  �  �  L    �  �  Q    �  ;  	C  	2  	?  	o  	z  	f  	K  	$  �  �  ]  �  �    �  6  �  r    �  �  �  �  �  �  p  X  @  (    �  �  �  �  �  �  l  L     �  �  �  �  �  �  }  v  m  [  I  5  #    .  H  H  H  G  A  8  �  �  �  �  �  �  �    R  Y  l  �  �  �  �  �    ,  E  ]    �  �  �    B  j  �  �  �  �  �  b  =    �  s  �  _        �      �  �  �  �  �  �  �  k  G    �  �  w  8   �     �    V  �  �  �  �  �    W  !  �  >  
}  	�  �  �  m  �  n  j  f  b  ]  V  O  H  @  7  #    �  �  �  �  �  �  �  �  	�    �  !  n  �  �  �  �  ^    �  C  
�  	�  �  �  �  F  	  b  �  �    m  �  �  �  �  �  S  :  #    �  g  �  �  �    �  �  �  �  �  �  �  �  t  d  Q  5    �  �  c    �  '   �        �  �  �  �  �  m  F    �  �  �  �  �  9  �  "   l  h  �  �  �  �  �  �  �  �  a  %  �  �  a    �  b    �  >  <  K  H  ;  $  	  �  �  �  b  N  /    �  �  8  �  O  �  �  �  �  �  	  ,  0  ,  !    �  �  �  S  �  �  !  �  3  <  �    C  b  q  z  }  t  Z  9  3  "        �  �  �  �  ^  �  �  �  �  �  �  q  _  G  -      �  �  �  �  o  G    �  �  �  �  S  h  �  �  y  c  @    �  \  �  ;  �  l  K  "  �  �  �  �  �  x  Y  R  7    �  �  �  �  S    �  �  j  J  5  �  S  U  V  X  Y  [  ]  [  X  U  R  O  L  K  P  U  Z  _  c  h  n  c  X  L  ?  2  "    �  �  �  �  �  �  y  ^  B  '     �    �  �  �  �  t  Z  D  /    
  �  �  �  �  �  �  k    �  �  �  �  �  �  �  |  o  b  V  K  B  9  /  &    �  �  �  [  �    P  �  �  �  	  �  �  �  �  �  L  �  �  �  =  �  i  m  �  	"  	N  	m  	y  	�  	�  	}  	r  	\  	=  	  �  j    u  �  �  C  �  Y  W  L  =  (      �  �  �  �  �  �  �  s  4  �  ~    �  .  H  i  n  m  x  t  b  E    �  `    �  '  �  Z  �  �  �  �  j  T  >  *    
     �  �  �  �  �  �  y  a  D  #  �  �  �    z  l  Z  G  &  �     B  9  1  &      �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  w  l  b  W  N  D  :  1  '      H    �  �  �  �  w  W  4  �  �  c  �  �  
  }  �  [  �  ~  o  `  O  =  *    �  �  �  �  �  V  �  �  �  4  �  �  =  �  �  �  �  �  r  P  ,    �  �  �  c  9    �  �  �  \  .  �  �  �  �  �  �  �  �  �  �  �  o  N  '     �  �  s  3   �  �  �  �  �  �  �  �  t  D    �  �  K    �  �  �  M  �  �  O  ,    �  �  �  S    �  �  T    �  Z    �  Q  �  �  o  y  O  %  �  �  �  �  �  n  S  7    �  �  �  Z  '  �  �  �  �  �  i  N  0    �  �  �  g  5    �  �  o  B    �  �  �