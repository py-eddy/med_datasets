CDF       
      obs    ?   NbSample         	   track_extra_variables         Uheight_max_speed_contour,height_external_contour,height_inner_contour,lon_max,lat_max      track_array_variables               array_variables       Dcontour_lon_e,contour_lat_e,contour_lon_s,contour_lat_s,uavg_profile   Metadata_Conventions      Unidata Dataset Discovery v1.0     comment       !Surface product; mesoscale eddies      framework_used        *https://github.com/AntSimi/py-eddy-tracker     framework_version         v3.4.0+24.ge934346     standard_name_vocabulary      HNetCDF Climate and Forecast (CF) Metadata Convention Standard Name Table   rotation_type                     	amplitude                   comment       �Magnitude of the height difference between the extremum of SSH within the eddy and the SSH around the effective contour defining the eddy edge     	long_name         	Amplitude      units         m      scale_factor      ?PbM���   
add_offset               min       ?`bM���   max       ?�p��
=q      �  �   effective_area                  comment       -Area enclosed by the effective contour in m^2      	long_name         Effective area     units         m^2    min       M��   max       Pܒ[      �  �   effective_contour_height                comment       )SSH filtered height for effective contour      	long_name         Effective Contour Height   units         m      min       ��   max       =��      �  �   effective_contour_latitude                     axis      X      comment       Latitudes of effective contour     	long_name         Effective Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @EУ�
=q     	�   �   effective_contour_longitude                    axis      X      comment       #Longitudes of the effective contour    	long_name         Effective Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @v�z�G�     	�  *x   effective_contour_shape_error                   comment       EError criterion between the effective contour and its best fit circle      	long_name         Effective Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @/         max       @Q            �  4P   effective_radius                comment       DRadius of the best fit circle corresponding to the effective contour   	long_name         Effective Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @�p@          �  4�   inner_contour_height                comment       5SSH filtered height for the smallest detected contour      	long_name         Inner Contour Height   units         m      min       ��/   max       >��      �  5�   latitude                axis      Y      comment       &Latitude center of the best fit circle     	long_name         Eddy Center Latitude   standard_name         latitude   units         degrees_north      min       A���   max       B,��      �  6�   latitude_max                axis      Y      comment       Latitude of the inner contour      	long_name         Latitude of the SSH maximum    standard_name         latitude   units         degrees_north      min       A�O�   max       B,�      �  7�   	longitude                   axis      X      comment       'Longitude center of the best fit circle    	long_name         Eddy Center Longitude      standard_name         	longitude      units         degrees_east   min       @֣   max       C��      �  8�   longitude_max                   axis      X      comment       Longitude of the inner contour     	long_name         Longitude of the SSH maximum   standard_name         	longitude      units         degrees_east   min       @�Y   max       C���      �  9�   num_contours                comment       )Number of contours selected for this eddy      	long_name         Number of contours     min             max          �      �  :�   num_point_e                 description       8Number of points for effective contour before resampling   	long_name         &number of points for effective contour     units         ordinal    min             max          A      �  ;�   num_point_s                 description       4Number of points for speed contour before resampling   	long_name         "number of points for speed contour     units         ordinal    min             max          9      �  <�   
speed_area                  comment       )Area enclosed by the speed contour in m^2      	long_name         
Speed area     units         m^2    min       M��   max       P���      �  =�   speed_average                   comment       IAverage speed of the contour defining the radius scale “speed_radius”      	long_name         Maximum circum-averaged Speed      units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��u��"    max       ?��	k��      �  >�   speed_contour_height                comment       %SSH filtered height for speed contour      	long_name         Speed Contour Height   units         m      min       ��   max       >�R      �  ?�   speed_contour_latitude                     axis      X      comment       Latitudes of speed contour     	long_name         Speed Contour Latitudes    units         degrees_east   scale_factor      ?�z�G�{   
add_offset               min       @>���
=q   max       @E�G�z�     	�  @�   speed_contour_longitude                    axis      X      comment       Longitudes of speed contour    	long_name         Speed Contour Longitudes   units         degrees_east   scale_factor      ?�z�G�{   
add_offset        @f�        min       ?�G�z�    max       @v�z�G�     	�  Jx   speed_contour_shape_error                   comment       AError criterion between the speed contour and its best fit circle      	long_name         Speed Contour Shape Error      units         %      scale_factor      ?�         
add_offset               min       @!         max       @Q            �  TP   speed_radius                comment       ZRadius of the best fit circle corresponding to the contour of maximum circum-average speed     	long_name         Speed Radius   units         m      scale_factor      @I         
add_offset               min       @͗        max       @�_�          �  T�   time                axis      T      calendar      proleptic_gregorian    comment       Date of this observation   	long_name         Time   standard_name         time   units         days since 1950-01-01 00:00:00     min         F�   max         F�      �  U�   uavg_profile                   comment       fSpeed averaged values from the effective contour inwards to the smallest contour, evenly spaced points     	long_name         Radial Speed Profile   units         m/s    scale_factor      ?6��C-   
add_offset               min       ?��`A�7L   max       ?�'RT`�     �  V�         
         
            z   	         E   (            %                  C   	   	      8   	   D      2            	      0      "         ,               2         )      2   w   �   o      "            Nq��O��NmպN8�nO2ӽN�ԑOO��M��NK)�Pܒ[N�ܱN�N$KEPIn�O�uO��NF�N��yO�M�O��N��VO*�AN��TNK�P���O��N�b�O�PGP��O,��O�|
NC��PNwN�[�N%�QO	PCN<+�Og+�O��O�O�cSNDm�N�h�P��GN6�O��2N֚�N��Ps�N�*UNr+O�4�N�m�O� �O�3�P:o�O�iO�uOA��Nɨ�N��gN�.N�]���T���#�
�#�
��`B��o��o��o;o;D��;ě�<#�
<#�
<49X<49X<T��<e`B<u<�C�<�t�<�t�<���<���<�1<�9X<�9X<�j<�j<ě�<ě�<���<�`B<�=C�=C�=\)=\)=�P=��=�w='�=,1=,1=0 �=0 �=49X=49X=8Q�=@�=@�=@�=L��=P�`=Y�=]/=e`B=e`B=e`B=}�=}�=�o=�t�=��"())5>BNRQNEB=5)""""��������������������99<=EHNTUTH<99999999�������������������� �	"/4;<:88/."	 ��������������������Uanojz�������zqn`bUUtst������ttttttttttt���������������������������)NdhgPB5����������������������� #$').#! #/40/*##!!!!!!!!`[[`gt�����������tg` #/<DHNTWPH</#
 )*6BOXZOMCB6-)"��������������������������������������������������
���������������������������������������������
#/<CHIH?3*&#
#'/464/# uvz�������uuuuuuuuuu����3BPY^[P5)����	
#0;;630,#
		���������������������������

�����������)BNWchi_N5)��qt���������������tq��
#/HTSTNG</#
������������������������
3:6/
�����{�������������{{{{{{������������������������%)))���4/+.58BCDBB544444444����)59BDB5.)�������������������������������������������������	��������!"#(0:<?<<0#!!!!!!!!cjnv{�������{nccccccgmz��������������zng������������������������)6?A@=6)������������������=<=ABMOS[\^][TOB====���-BGFHKLHA5)��LKNS[gttxwtgg[QNLLLL.,,/<=?ACB=<;84/....*#)*)/;Tmu{zsmaTH;0*�����

�����������
�����������������	�����<8:BN[t��������g[NB<��������#)/)$����16BO[htwxwth[OHB>861{z�����������������{\WVZanoswz{{znaa\\\\��������������������`XZakmonma``````````aUMHFFHUWantwngaaaaa����������������������ŹŸŹż���������ƺ������������Ǻ�������������������������D�EEEEE"EEED�D�D�D�D�D�D�D�D�D�D������ʼʼּؼּʼ������������������������H�T�_�a�e�l�o�m�e�a�T�H�/�"�"�/�;�=�D�H�C�O�P�V�\�b�b�\�O�C�8�6�5�3�6�?�C�C�C�C��������������������������������������������������������������������������������¿��������¿¿²«­²³¿¿¿¿¿¿¿¿�����0�I�{�~�u�h�U�<�0�������ķįĻĹ���������'�)�-�'��������������A�J�N�Z�g�l�g�Z�N�A�:�@�A�A�A�A�A�A�A�A�����������������������|������������������/�H�a�i�l�k�]�T�H�;�"����������������������� � �������������������뾱���ʾ׾׾ھپؾ׾ʾ¾�����������������F=FJFVFcFlFcFZFVFJF=F5F6F=F=F=F=F=F=F=F=�����ʾ׾ݾ׾Ҿʾ��������������������������������û˻ʻĻ����������l�R�S�Y�l�z�����������������������z�m�h�d�b�l�r�z�����/�;�H�R�T�Z�T�K�H�;�0�/�,�-�/�/�/�/�/�/�m�i�m���������������������z�m�a�O�M�T�m�/�7�;�E�;�5�/�"�����"�,�/�/�/�/�/�/���ûлջԻлû������������������������������#�)���������ƚƎ�u�J�E�L�W�h�u�̼����������������߼�����(�4�A�M�P�M�J�I�M�A�4�(�%�����$�(�(�A�M�Z�f�s�x�x�n�\�M�A�4�����(�-�4�A������������������������������������޾Z�[�f�s���������������s�i�f�_�Q�V�U�Z�`�m�����������������m�`�T�L�H�G�I�O�T�`�h�tāāČā�t�h�a�^�h�h�h�h�h�h�h�h�h�h�5�N�g�k�n�f�e�a�[�B�5�)�%��� ����5�@�M�P�Y�\�f�j�f�Y�M�@�=�<�8�@�@�@�@�@�@���!�(�4�(���������������H�U�a�i�m�l�c�a�U�J�H�C�=�<�6�;�<�G�H�H���
��#�)�#���
�����������������������M�Z�^�f�i�m�k�i�f�M�A�3�+�4�8�<�;�8�H�M������8�A�E�G�?�5�(���������������������ĽнֽѽĽ����������|�z���������a�m�}���y�h�a�T�H�;�/�#�����	��"�;�a�'�4�7�@�K�@�>�4�(�'��$�'�'�'�'�'�'�'�'�������������������������������������������������s�Z�5����8�D�V�s���������������������������~�������������������~�r�b�N�O�V�X�a�n�~�!�"�-�:�;�E�F�N�F�:�-�!������ �!�!�ܻ��������������ܻڻջܻܻܻܿ������ƿͿѿڿѿĿ��y�m�`�T�K�I�V�`�y���������
��������ݿݿܿݿ�����ǈǔǡǬǣǡǔǈ�{�o�b�_�b�o�{Ǆǈǈǈǈ�������
������������ĿĹĲĮİĴĿ�ؾ������ʾվϾʾ��������������������������������������������y�l�`�V�P�N�P�S�d�d��D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D}DD�D�D����6�G�S�W�U�K�6�%�������������������������ʼ�������ּʼ��������������������������������������r�\�W�\�f�r����������������������������������������޻����ûл׻ѻлû�����������������������ÇÓÜàìíìëåàÓÇÀ�z�w�x�z�{ÇÇŭŹ��������ŹŭŬŬŭŭŭŭŭŭŭŭŭŭ��������'�(�+�+�(�������� n ; : Z H > 7 I J " z l h . 1 X j 9 g O & � B 5 3 ) 6 $ H R   C - ] V 2 O X > _ 2 1 g < = 2 D 9 - # � < e B < & N . 0 Y  J P    �  @  �  u  �  �  �  /  p  �  �  e  h  c  +  ]  h  �  b  �  �  �  �  i  1  "    P  �  �  �  Y  F  �  B  6  d  	  _  =  C  [  �  �  V  �  �  �  �  �  �  �  �  �  T    �    �  �  �  7  ¼�/;ě�%   ��`B;D��<t�;��
:�o<�t�>   <e`B<D��<e`B=��
=P�`<ě�<���<��
=]/<�<�1<�h<ě�=t�=�9X<�<��=8Q�=���=+=�v�=#�
=���=�w='�=L��=49X=m�h=�{=�o=���=T��=]/=� �=<j=���=e`B=m�h=ě�=T��=P�`=�Q�=ix�=��`>0 �>��>+=� �=ě�=��
=��
=��P=�9XBl&B")�B1�B"fA���B,�B��B��B GBv�B"DmB��B�B
��B>�B��B|�B"�HB"w�B&�B��B1�B�5B��B�hB%�B 1B#d�BٍBB��B	{B1�B B�FB�RBX"B*LB�jB"�uB��B%��B(�iB�fBF�BJ>B3�B��BpbB	�B�7A�L�BϸB,��B�B	�B��B]B1�B��B!}�A���B�B�VB"?�B?�B"nA�O�B��B�}B�1B ?
B��B"A�B�BB��B
@`B��B��BEB#=�B"��B�IBJ,B=4B�ZBB<B%	�B�)B#�	B�B@�B>%B?B:�B��B�BB��B@!B?vB�'B"�AB��B%�&B(��BAB<�B��B��B��BCNB	;>BΊA�q!BJ�B,�B�B	}�B�B?�B<>B�,B!�A��-B�CA��O@=	C�[@��A��6BA��KA�&@A�TA�K�@�+�A���A��A�E�A���AO�C��ANд@��A�3�A���A��$A��@�Z�B;�A6�A9iA;̄A��vAB�Al A�ooA��V@֟�A�aA�YjA��A><�A�(A"�^A� �@�u*@[�A���A�K@֣@t:G@��Ap!oA�:�B��A��AMj^A��C��MA��6@���@饭Aэ�@���A�C^A��qA4�6A�~�@w�C�X@�:�A�ƩB@#A��A��A�|A鄃@�$A�Z#A�f�A���A҄AO-�C���AMo@�@�A��[A���A�:A��?@���B�A^ZA8��A<qtA�y�AB��Al��A܀.A���@��UA��A�5A�{�A>��A�~MA#�A��@�F>@[#�A�{�A�Tt@�Y@l�@���Ap�.A�B�qA�AO0BA�wC���A�3�@���@�ֶAы@��A���A��GA3�m                              {   	         F   )            &                  C   	   	      8   
   E      2            
      0      #         ,                2         )      2   w   �   p      #                                 !         A            +               '                  ;            '            %                        %         ;      !         %         !      #      )   #                                                )                           '                  9            #                                    #         7      !                                                   Nq��N�l�NmպN8�nO2ӽN�K�O��M��NK)�P=E8N�ܱN�N$KEO�
�O-M|N��NF�N��yO�M�N˷�N��VNāCN��TN��P���N�ʿN�b�O�2O�x(O,��O~y�NC��O��N�[�N%�QO	PCN<+�N�FCO�aBO�O�NDm�N�h�P�>N6�O��2N֚�N�;_O��+N�*UNr+O;'aN�m�OW�O$�O���O�ۙO�uO9~�Nɨ�N��gN�.N�]  s  d  �  �     Q  �  r  Y  &  �  7  n  y  �  �  �  �  x  =    �      �  G  �  �  z  �  	  5  �  �  �  9  =  	  P  �  q  �  �  �  �  S  �  .  �  :  [  �  �  �  �  �  y  K  �  �  �    4���ě��#�
�#�
��`B%   %   ��o;o=P�`;ě�<#�
<#�
='�<�9X<u<e`B<u<�C�<�1<�t�<�1<���<�j<���<ě�<�j<���=o<ě�='�<�`B=\)=C�=C�=\)=\)=49X='�=�w=0 �=,1=,1=49X=0 �=49X=49X=<j=m�h=@�=@�=��=P�`=�O�=�^5>�R=��=e`B=�%=}�=�o=�t�=��"())5>BNRQNEB=5)""""��������������������99<=EHNTUTH<99999999�������������������� �	"/4;<:88/."	 ��������������������innrnz����������zunitst������ttttttttttt���������������������������%5>EDA5)����������������������� #$').#! #/40/*##!!!!!!!!fgit�������������tkf#/<FHNQH<2/(# )16BFOJB?6)��������������������������������������������������
���������������������������������������������##/<FE<;//&##'/464/# xw�������xxxxxxxxxx���)5BNWZ[YN5)����

"#-05200'#


�����������������������������

��������)BIR_cc[N5)��qt���������������tq	
#/=HJLJH</#
�����������������������
$+.-$
�����{�������������{{{{{{������������������������%)))���4/+.58BCDBB544444444��&)5865)�������������������������������������������������
	��������!"#(0:<?<<0#!!!!!!!!cjnv{�������{ncccccchlz��������������znh������������������������)6?A@=6)������������������=<=BBNOP[\][[SOB====���)5>@ACC>5)�LKNS[gttxwtgg[QNLLLL.,,/<=?ACB=<;84/....857:;HNTakqpmdaTHG=8�����

����������������������������������� 
	�����MIHJP[gt��������tg[M�������������16BO[htwxwth[OHB>861}{�����������������}\WVZanoswz{{znaa\\\\��������������������`XZakmonma``````````aUMHFFHUWantwngaaaaa����������������������ŹŸŹż���������ƺ���������������������������������������D�EEEEE"EEED�D�D�D�D�D�D�D�D�D�D������ʼʼּؼּʼ������������������������H�T�_�a�e�l�o�m�e�a�T�H�/�"�"�/�;�=�D�H�C�D�O�U�\�a�a�\�O�C�:�6�6�5�6�C�C�C�C�C��������������������������������������������������������������������������������¿��������¿¿²«­²³¿¿¿¿¿¿¿¿������0�I�U�W�S�I�<�0�#�
���������������������'�)�-�'��������������A�J�N�Z�g�l�g�Z�N�A�:�@�A�A�A�A�A�A�A�A�����������������������|�����������������"�/�;�O�S�R�N�H�;�/�"�	�������������"������������������������������뾱���¾ʾӾ׾ؾ׾׾ʾ�������������������F=FJFVFcFlFcFZFVFJF=F5F6F=F=F=F=F=F=F=F=�����ʾ׾ݾ׾Ҿʾ��������������������������������û˻ʻĻ����������l�R�S�Y�l�z���z���������������������z�m�m�i�m�q�x�z�z�/�;�H�R�T�Z�T�K�H�;�0�/�,�-�/�/�/�/�/�/�m�z���������������������z�y�m�d�m�m�m�m�/�7�;�E�;�5�/�"�����"�,�/�/�/�/�/�/���ûлѻһлû���������������������������������������ƳƚƎ�u�N�H�N�\�hƁ�̼��������������������������(�4�A�M�P�M�J�I�M�A�4�(�%�����$�(�(�4�A�M�Z�d�f�s�t�t�i�Z�M�A�4�!���&�2�4������������������������������������޾Z�[�f�s���������������s�i�f�_�Q�V�U�Z�`�m�����������������y�m�`�S�O�N�R�T�[�`�h�tāāČā�t�h�a�^�h�h�h�h�h�h�h�h�h�h�)�5�N�f�j�c�b�[�N�B�5�)��������)�@�M�P�Y�\�f�j�f�Y�M�@�=�<�8�@�@�@�@�@�@���!�(�4�(���������������H�U�a�i�m�l�c�a�U�J�H�C�=�<�6�;�<�G�H�H���
��#�)�#���
�����������������������M�Z�b�f�i�g�f�f�_�Z�M�J�C�C�A�@�A�H�M�M������4�C�E�A�;�5�(��������������������ĽнֽѽĽ����������|�z���������a�m�|�~�z�f�a�T�H�;�/�%��	����"�;�a�'�4�7�@�K�@�>�4�(�'��$�'�'�'�'�'�'�'�'�������������������������������������������������g�Z�5�$�� �9�E�W�s���������������������������~�������������������~�r�b�N�O�V�X�a�n�~�!�"�-�:�;�E�F�N�F�:�-�!������ �!�!�ܻ��������������ܻڻֻܻܻܻܿ����������ÿĿÿ������y�m�`�V�Q�V�`�m���������
��������ݿݿܿݿ�����ǈǔǡǬǣǡǔǈ�{�o�b�_�b�o�{Ǆǈǈǈǈ��������� ���������������ĿĽĻľĿ���ؾ������ʾվϾʾ��������������������������y���������������������y�l�`�Z�W�\�`�l�yD�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�D�����)�6�?�C�B�<�6�)�������������������ʼּ������ּʼ��������������������������������������r�\�W�\�f�r������������������������������������������޻����ûл׻ѻлû�����������������������ÇÓÜàìíìëåàÓÇÀ�z�w�x�z�{ÇÇŭŹ��������ŹŭŬŬŭŭŭŭŭŭŭŭŭŭ��������'�(�+�+�(�������� n 4 : Z H @ * I J  z l h ' & K j 9 g U & f B ? 0 ( 6  F R  C # ] V 2 O P ; _ ) 1 g : = 2 D = * # � 2 e 6 4  9 . / Y  J P    �  �  �  u  �  �  Q  /  p    �  e  h  z  q    h  �  b    �    �  ?  �  �    �  &  �  �  Y  �  �  B  6  d    "  =    [  �  �  V  �  �  �  �  �  �  �  �  �  c  E  i    �  �  �  7  �  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  F�  s  k  c  [  S  K  C  9  -  !    	  �  �  �  �  �  �  �  �  �    (  <  L  Y  `  d  ]  O  ;  #    �  �  }  T  *  �  �  �  �  �  �  �    b  D  #  �  �  �  �  W  )  �  �  �  |  R  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  z     �  �  �  �  �  �  e  M  ;  +      �  �  �  m  <    �  P  P  P  M  J  D  >  8  2  ,  (  $          �  �  �  �  n  u  {  �  �  �    z  s  l  d  Z  P  D  3  "     �   �   �  r  n  j  e  a  \  X  T  O  K  I  I  I  H  H  H  H  H  H  H  Y  �  �  H  �  �  d  �  #  Z  �  �  �  	  	D  	t  	�  	�  
  
3  &  �  �  [  �  �    %  "    �  �  �  =  �  _  �  c  {  �  �  �  �  �  �        �  �  �  �  �  �  |  a  F  *    �  7  -  $        �  �  �  �  �  �  �  �  �  �  �  �  �  �  n  b  V  J  >  1    
  �  �  �  �  �  �  �  j  R  :  "    c  �  �  �    6  S  l  x  t  _  =    �  R  �    L  u   �  �    J  r  �  �  �  w  T  %  �  �  l    �    r  �  4  �  �  �  �  �  �  �  �  �  �  �  �  �  y  [  :  
  �  �  {  N  �  �  �  }  z  w  t  q  k  `  V  L  =  +      �  �  �  �  �  �  {  t  n  g  `  Y  P  E  ;  0  +  *  )  (       �   �  x  o  ^  J  4      �  �  �  �  �  c      B  2    �  �      (  2  9  <  7  +      �  �  �  ~  W  ,  �  �  M   �        	     �   �   �   �   �   �   �   �   �   �   �   �   �   �   �  s  �  �  �  �  �  �  �  �  �  �  �  �  �  l  M  $  �  �  �          �  �  �  �  �  �  �  �  �  }  ^  ?  1  '      �  �        !        �  �  �  �  g  ,  �  �  l  (  �  �  �  �  �  ^  /  �  �  ^    �  �  J    �  �  �  F  �  �  :  ?  D  F  G  G  E  D  <  4  *         �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  �  y  [  >    �  �  f  /  y  �  �  �  �  �  z  i  T  =  #    �  �  �  X  +  �  �  b    V  s  z  t  a  I  >  (    �  �  E  �  �    �  K    �  �  x  o  q  s  r  p  f  X  H  4    	  �  �  �  �  �  o  X  �  �  	  	  	  	  	  �  �  �  h  3  �  �  @  �  �  �  �  �  5      �  �  �  �  �  f  P  7    �  �  �  \  )  �  �  �  �  �  �  �  �  �  s  N  '  �  �  e  
  �  9  �  7  �      �  �  �  �  ~  �  �  �  �  �  �  �      &  7  D  Q  ^  k  �  �  �  �  �  �  y  `  H  -    �  �  �  �  m  D     �   �  9  /       �  �  �  �  �  m  O  0    �  �  �  �  P    �  =  *      �  �  �  �  j  A    �  �  �  V     �  �  h  )  �  �  �  �  �  �  �    �  �  �  U    �  �  B      �  �  +  O  N  F  <  *    �  �  �  o  7  �  �  O  �    9  S  ]  �  �  q  e  K    �  �  �  s  7  �  �  ;  �  r     |  �  2  a  i  R  4    �  �  �  �  a  1  �  �  j    �  G  �  v     �  �  �  �  �  �  �  ~  o  ^  K  7  #    �  �  �  �  r  /  �  �  s  ]  F  /    �  �  �  �  |  W  :  /  �  �  O  �  L  �  �  �  �  ]  %  �  �  G  �  �  5  �  h    �  U  �  m   �  �  ~  z  v  r  n  j  d  ]  V  O  H  A  6  %      �  �  �  S  N  A  -    �  �  �  n  =    �  b    �  k  �  w  �  �  �  �  �  �  �  �  �  �  �  k  Q  2    �  �  �  d    �  "  !  *  ,  '      �  �  �  �  �  g  G  &    �  �  �  Y  "  Q  �  �  �  �  �  �  �  �  �  h  +  �  r  �  l  �  B  �   �  :  /  %        �  �  �  �  �  �  �  �  �  �  w  i  [  N  [  X  U  R  O  L  E  >  7  0  #    �  �  �  �  �  z  \  =  �  &  k  �  �  �  �  �  �  �  �  ?  �  �  @  �  q  
  �  h  �  �  �  �  |  h  S  B  2  #    �  �  �  �  �  �  k  >    T  m  �  �  �  �  �  �  �  �  �  �  �  x  B  �  �  	  ~  a  L  �    K  o  �  �  �  v  S  &  �  �  �  !  (  �  
�  �  �  �  �  q  <  �  V  �  �  �  �  �  d  �  �  �  ?  u  �  ;  ;  .  �  !  c  x  v  g  E    �  n    �  �  3  
6  	  �  �  �  K  ?  '  	  �  �  �  }  T  #  �  �  ^    �  X    �  2  �  �  �  �  �  �  �  �  v  V  +  �  �  Y  �  �  !  �  O  �  �  �  �  s  W  =  "    �  �  �  �  �  �  �  �  �  y  4  �  �  �  �  �  q  L  &     �  �  �  T  $  �  �  w  .  �  �  0  �      �  �  �  �  �  �  �  ~  l  Y  E  2      �  �  �  �  4  .  )  #        �  �  �  �  �  �  �  �  �  �  k  U  ?